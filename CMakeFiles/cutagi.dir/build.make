# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Users/anaconda/anaconda3/envs/test10/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /Users/anaconda/anaconda3/envs/test10/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jaebeomlee/Documents/Codes/cuTAGI

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jaebeomlee/Documents/Codes/cuTAGI

# Include any dependencies generated for this target.
include CMakeFiles/cutagi.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cutagi.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cutagi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cutagi.dir/flags.make

CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.o: CMakeFiles/cutagi.dir/flags.make
CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.o: src/python_api_cpu.cpp
CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.o: CMakeFiles/cutagi.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jaebeomlee/Documents/Codes/cuTAGI/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.o -MF CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.o.d -o CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.o -c /Users/jaebeomlee/Documents/Codes/cuTAGI/src/python_api_cpu.cpp

CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jaebeomlee/Documents/Codes/cuTAGI/src/python_api_cpu.cpp > CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.i

CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jaebeomlee/Documents/Codes/cuTAGI/src/python_api_cpu.cpp -o CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.s

# Object files for target cutagi
cutagi_OBJECTS = \
"CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.o"

# External object files for target cutagi
cutagi_EXTERNAL_OBJECTS =

cutagi.cpython-310-darwin.so: CMakeFiles/cutagi.dir/src/python_api_cpu.cpp.o
cutagi.cpython-310-darwin.so: CMakeFiles/cutagi.dir/build.make
cutagi.cpython-310-darwin.so: libcutagi_lib.a
cutagi.cpython-310-darwin.so: CMakeFiles/cutagi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/jaebeomlee/Documents/Codes/cuTAGI/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module cutagi.cpython-310-darwin.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cutagi.dir/link.txt --verbose=$(VERBOSE)
	/Library/Developer/CommandLineTools/usr/bin/strip -x /Users/jaebeomlee/Documents/Codes/cuTAGI/cutagi.cpython-310-darwin.so

# Rule to build all files generated by this target.
CMakeFiles/cutagi.dir/build: cutagi.cpython-310-darwin.so
.PHONY : CMakeFiles/cutagi.dir/build

CMakeFiles/cutagi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cutagi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cutagi.dir/clean

CMakeFiles/cutagi.dir/depend:
	cd /Users/jaebeomlee/Documents/Codes/cuTAGI && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jaebeomlee/Documents/Codes/cuTAGI /Users/jaebeomlee/Documents/Codes/cuTAGI /Users/jaebeomlee/Documents/Codes/cuTAGI /Users/jaebeomlee/Documents/Codes/cuTAGI /Users/jaebeomlee/Documents/Codes/cuTAGI/CMakeFiles/cutagi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cutagi.dir/depend

