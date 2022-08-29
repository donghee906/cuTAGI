///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_backward_cpu.cpp
// Description:  Long-Short Term Memory (LSTM) state backward pass in TAGI
//               (cpu version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2022
// Updated:      August 29, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#include "../include/lstm_feed_backward_cpu.h"

void lstm_delta_mean_var_z(std::vector<float> &Sz, std::vector<float> &mw,
                           std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
                           std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
                           std::vector<float> &Jc_ga, std::vector<float> &mo_ga,
                           std::vector<float> &Jo_ga,
                           std::vector<float> &mc_prev, std::vector<float> &mca,
                           std::vector<float> &Jca, std::vector<float> &delta_m,
                           std::vector<float> &delta_S, int z_pos_i,
                           int z_pos_o, int z_pos_o_lstm, int w_pos_f,
                           int w_pos_i, int w_pos_c, int w_pos_o, int no,
                           int ni, int n_seq, int B,
                           std::vector<float> &delta_mz,
                           std::vector<float> &delta_Sz)
/*Compute the updated quatitites of the mean of the hidden states for lstm
   layer*/
{
    float sum_mf, sum_mi, sum_mc, sum_mo, sum_Sz;
    float Czz_f, Czz_i, Czz_c, Czz_o;
    int k, m, i;
    // TODO Forget about number of sequences
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < n_seq; y++) {
            for (int z = 0; z < ni; z++) {
                sum_mf = 0;
                sum_mi = 0;
                sum_mc = 0;
                sum_mo = 0;
                sum_Sz = 0;
                for (int j = 0; j < no; j++) {
                    k = j + x * no * n_seq + y * no + z_pos_o_lstm;
                    i = j + x * no * n_seq + y * no + z_pos_o;
                    // Forget gate
                    Czz_f = Jca[k] * mo_ga[k] * Jf_ga[k] *
                            mw[(ni + no) * j + z + w_pos_f] * mc_prev[k];
                    sum_mf += Czz_f * delta_m[i];

                    // Input gate
                    Czz_i = Jca[k] * mo_ga[k] * Ji_ga[k] *
                            mw[(ni + no) * j + z + w_pos_i] * mc_ga[k];
                    sum_mi += Czz_i * delta_m[i];

                    // Cell state gate
                    Czz_c = Jca[k] * mo_ga[k] * Jc_ga[k] *
                            mw[(ni + no) * j + z + w_pos_c] * mi_ga[k];
                    sum_mc += Czz_c * delta_m[i];

                    // Output gate
                    Czz_o = Jo_ga[k] * mw[(ni + no) * j + z + w_pos_o] * mca[k];
                    sum_mo += Czz_o * delta_m[i];
                    sum_Sz +=
                        powf(Czz_f + Czz_i + Czz_c + Czz_o, 2) * delta_S[i];
                }

                // Updating quantities
                m = x * ni * n_seq + y * ni + z;
                delta_mz[m] =
                    (sum_mf + sum_mi + sum_mc + sum_mo) * Sz[m + z_pos_i];
                delta_Sz[m] = Sz[m + z_pos_i] * sum_Sz * Sz[m + z_pos_i];
            }
        }
    }
}

void lstm_delta_mean_var_w(std::vector<float> &Sw, std::vector<float> &mha,
                           std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
                           std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
                           std::vector<float> &Jc_ga, std::vector<float> &mo_ga,
                           std::vector<float> &Jo_ga,
                           std::vector<float> &mc_prev, std::vector<float> &mca,
                           std::vector<float> &Jc, std::vector<float> &delta_m,
                           std::vector<float> &delta_S, int z_pos_o,
                           int z_pos_o_lstm, int w_pos_f, int w_pos_i,
                           int w_pos_c, int w_pos_o, int no, int ni, int n_seq,
                           int B, std::vector<float> &delta_mw,
                           std::vector<float> &delta_Sw)
/*Compute updating quantities of the weight parameters for lstm layer */
{
    float sum_mf, sum_Sf, Cwa_f, sum_mi, sum_Si, Cwa_i, sum_mc, sum_Sc, Cwa_c,
        sum_mo, sum_So, Cwa_o;
    int k, m, l, i;
    for (int row = 0; row < (ni + no); row++) {
        for (int col = 0; col < no; col++) {
            sum_mf = 0;
            sum_Sf = 0;
            sum_mi = 0;
            sum_Si = 0;
            sum_mc = 0;
            sum_Sc = 0;
            sum_mo = 0;
            sum_So = 0;
            for (int x = 0; x < B; x++) {
                for (int y = 0; y < n_seq; y++) {
                    k = col + y * n_seq + no * n_seq * x + z_pos_o_lstm;
                    i = col + y * n_seq + no * n_seq * x + z_pos_o;
                    l = row + y * (ni + no) + (ni + no) * n_seq * x;

                    // Forget gate
                    Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k] * mha[l];
                    sum_mf += Cwa_f * delta_m[i];
                    sum_Sf += Cwa_f * delta_S[i] * Cwa_f;

                    // Input gate
                    Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k] * mha[l];
                    sum_mi += Cwa_i * delta_m[i];
                    sum_Si += Cwa_i * delta_S[i] * Cwa_i;

                    // Cell state gate
                    Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k] * mha[l];
                    sum_mc += Cwa_c * delta_m[i];
                    sum_Sc += Cwa_c * delta_S[i] * Cwa_c;

                    // Output gate
                    Cwa_o = Jo_ga[k] * mca[k] * mha[l];
                    sum_mo += Cwa_o * delta_m[i];
                    sum_So += Cwa_o * delta_S[i] * Cwa_o;
                }
            }
            // Updating quantities for weights
            m = col * (ni + no) + row;
            delta_mw[m + w_pos_f] = sum_mf * Sw[m + w_pos_f];
            delta_Sw[m + w_pos_f] = Sw[m + w_pos_f] * sum_Sf * Sw[m + w_pos_f];

            delta_mw[m + w_pos_i] = sum_mi * Sw[m + w_pos_i];
            delta_Sw[m + w_pos_i] = Sw[m + w_pos_i] * sum_Si * Sw[m + w_pos_i];

            delta_mw[m + w_pos_c] = sum_mc * Sw[m + w_pos_c];
            delta_Sw[m + w_pos_c] = Sw[m + w_pos_c] * sum_Sc * Sw[m + w_pos_c];

            delta_mw[m + w_pos_o] = sum_mo * Sw[m + w_pos_o];
            delta_Sw[m + w_pos_o] = Sw[m + w_pos_o] * sum_So * Sw[m + w_pos_o];
        }
    }
}

void lstm_delta_mean_var_b(std::vector<float> &Sb, std::vector<float> &Jf_ga,
                           std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
                           std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
                           std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
                           std::vector<float> &mc_prev, std::vector<float> &mca,
                           std::vector<float> &Jc, std::vector<float> &delta_m,
                           std::vector<float> &delta_S, int z_pos_o,
                           int z_pos_o_lstm, int b_pos_f, int b_pos_i,
                           int b_pos_c, int b_pos_o, int no, int n_seq, int B,
                           std::vector<float> &delta_mb,
                           std::vector<float> &delta_Sb)
/*Compute updating quantities of the bias for the lstm layer */
{
    float sum_mf, sum_Sf, Cwa_f, sum_mi, sum_Si, Cwa_i, sum_mc, sum_Sc, Cwa_c,
        sum_mo, sum_So, Cwa_o;
    int k, l, i;
    for (int row = 0; row < no; row++) {
        sum_mf = 0;
        sum_Sf = 0;
        sum_mi = 0;
        sum_Si = 0;
        sum_mc = 0;
        sum_Sc = 0;
        sum_mo = 0;
        sum_So = 0;
        for (int x = 0; x < B; x++) {
            for (int y = 0; y < n_seq; y++) {
                k = row + y * n_seq + no * n_seq * x + z_pos_o_lstm;
                i = row + y * n_seq + no * n_seq * x + z_pos_o;

                // Forget gate
                Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k];
                sum_mf += Cwa_f * delta_m[i];
                sum_Sf += Cwa_f * delta_S[i] * Cwa_f;

                // Input gate
                Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k];
                sum_mi += Cwa_i * delta_m[i];
                sum_Si += Cwa_i * delta_S[i] * Cwa_i;

                // Cell state gate
                Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k];
                sum_mc += Cwa_c * delta_m[i];
                sum_Sc += Cwa_c * delta_S[i] * Cwa_c;

                // Output gate
                Cwa_o = Jo_ga[k] * mca[k];
                sum_mo += Cwa_o * delta_m[i];
                sum_So += Cwa_o * delta_S[i] * Cwa_o;
            }
        }
        // Updating quantities for biases
        delta_mb[row + b_pos_f] = sum_mf * Sb[row + b_pos_f];
        delta_Sb[row + b_pos_f] =
            Sb[row + b_pos_f] * sum_Sf * Sb[row + b_pos_f];

        delta_mb[row + b_pos_i] = sum_mi * Sb[row + b_pos_i];
        delta_Sb[row + b_pos_i] =
            Sb[row + b_pos_i] * sum_Si * Sb[row + b_pos_i];

        delta_mb[row + b_pos_c] = sum_mc * Sb[row + b_pos_c];
        delta_Sb[row + b_pos_c] =
            Sb[row + b_pos_c] * sum_Sc * Sb[row + b_pos_c];

        delta_mb[row + b_pos_o] = sum_mo * Sb[row + b_pos_o];
        delta_Sb[row + b_pos_o] =
            Sb[row + b_pos_o] * sum_So * Sb[row + b_pos_o];
    }
}

void lstm_state_update_cpu(Network &net, NetState &state, Param &theta,
                           DeltaState &d_state, int l)
/*Update lstm's hidden states*/
{
    // Initialization
    int ni = net.nodes[l];
    int no = net.nodes[l + 1];
    int z_pos_i = net.z_pos[l];
    int z_pos_o = net.z_pos[l + 1];
    int z_pos_o_lstm = net.z_pos_lstm[l + 1];
    int w_pos_f, b_pos_f, w_pos_i, b_pos_i, w_pos_c, b_pos_c, w_pos_o, b_pos_o;
    int ni_c = ni + no;

    w_pos_f = net.w_pos[l];
    b_pos_f = net.b_pos[l];
    w_pos_i = net.w_pos[l] + ni_c * no;
    b_pos_i = net.b_pos[l] + no;
    w_pos_c = net.w_pos[l] + 2 * ni_c * no;
    b_pos_c = net.b_pos[l] + 2 * no;
    w_pos_o = net.w_pos[l] + 3 * ni_c * no;
    b_pos_o = net.b_pos[l] + 3 * no;

    lstm_delta_mean_var_z(
        state.Sz, theta.mw, state.lstm.Jf_ga, state.lstm.mi_ga,
        state.lstm.Ji_ga, state.lstm.mc_ga, state.lstm.Jc_ga, state.lstm.mo_ga,
        state.lstm.Jo_ga, state.lstm.mc_prev, state.lstm.mca, state.lstm.Jca,
        d_state.delta_m, d_state.delta_S, z_pos_i, z_pos_o, z_pos_o_lstm,
        w_pos_f, w_pos_i, w_pos_c, w_pos_o, no, ni, net.input_seq_len,
        net.batch_size, d_state.delta_mz, d_state.delta_Sz);
}

void lstm_parameter_update_cpu(Network &net, NetState &state, Param &theta,
                               DeltaState &d_state, DeltaParam &d_theta, int l)
/*Update lstm's parameters*/
{
    // Initialization
    int ni = net.nodes[l];
    int no = net.nodes[l + 1];
    int z_pos_i = net.z_pos[l];
    int z_pos_o = net.z_pos[l + 1];
    int z_pos_o_lstm = net.z_pos_lstm[l + 1];
    int w_pos_f, b_pos_f, w_pos_i, b_pos_i, w_pos_c, b_pos_c, w_pos_o, b_pos_o;
    int ni_c = ni + no;

    w_pos_f = net.w_pos[l];
    b_pos_f = net.b_pos[l];
    w_pos_i = net.w_pos[l] + ni_c * no;
    b_pos_i = net.b_pos[l] + no;
    w_pos_c = net.w_pos[l] + 2 * ni_c * no;
    b_pos_c = net.b_pos[l] + 2 * no;
    w_pos_o = net.w_pos[l] + 3 * ni_c * no;
    b_pos_o = net.b_pos[l] + 3 * no;

    // Concatenate the hidden states from the previous time step and activations
    // from the previous layer
    cat_activations_and_prev_states(state.ma, state.lstm.mh_prev, ni, no,
                                    z_pos_i, z_pos_o_lstm, state.lstm.mha);

    lstm_delta_mean_var_w(
        theta.Sw, state.lstm.mha, state.lstm.Jf_ga, state.lstm.mi_ga,
        state.lstm.Ji_ga, state.lstm.mc_ga, state.lstm.Jc_ga, state.lstm.mo_ga,
        state.lstm.Jo_ga, state.lstm.mc_prev, state.lstm.mca, state.lstm.Jca,
        d_state.delta_m, d_state.delta_S, z_pos_o, z_pos_o_lstm, w_pos_f,
        w_pos_i, w_pos_c, w_pos_o, no, ni, net.input_seq_len, net.batch_size,
        d_theta.delta_mw, d_theta.delta_Sw);

    lstm_delta_mean_var_b(
        theta.Sb, state.lstm.Jf_ga, state.lstm.mi_ga, state.lstm.Ji_ga,
        state.lstm.mc_ga, state.lstm.Jc_ga, state.lstm.mo_ga, state.lstm.Jo_ga,
        state.lstm.mc_prev, state.lstm.mca, state.lstm.Jca, d_state.delta_m,
        d_state.delta_S, z_pos_o, z_pos_o_lstm, b_pos_f, b_pos_i, b_pos_c,
        b_pos_o, no, net.input_seq_len, net.batch_size, d_theta.delta_mb,
        d_theta.delta_Sb);
}
