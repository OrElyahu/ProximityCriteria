# ParallelComputation
A project made with MPI, OpenMP and CUDA on Linux

# Problem definition:
  We will say that some point P, from the set (given in a .txt), satisfies a 'Proximity Criteria' if:
  There exist at least K points in the set with a distance from the point P, less than a given value D.
  Given a value of parameter t, we want to find if there exist at least 3 points that satisfies the Proximity Criteria.

# Efficiency:
  This project supposed to run on two different machines, each of them has at least 4 cores.
  Due to the fact of 8 cores, we parallel this project to be much faster, comparing running sequentially.

# Compile:
  First extract the "Common.zip" to this folder.
  Then, you need to compile it within the terminal, using command line: make
  Afterwards, should be a 'project' file to run the program.

# Running:
  In the folder, there are 2 files for examples, one named "Input.txt" and the other one "Input2.txt"
  The program takes the file with name "Input.txt" and makes an "Output.txt" file as the required. 
  (Case you want to run "Input2.txt", rename it to "Input.txt")

  **Note**
	  This program runs on 2 different machines, therefore should be address the mf file to the wanted IPs.
	  To acheive so, can be running with command: hostname -I
	
   To run this program write : make runOn2e
