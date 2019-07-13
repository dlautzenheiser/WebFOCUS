# test program called by WebFOCUS

# for python system input arguments 
import sys

# function to show arguments
def show_arguments():
	print ('Running Python script named %s' % (sys.argv[0]))
	if len(sys.argv) > 1:
		# in addition to program name, at least one parameter was provided
		print ('#Arguments= ', len(sys.argv))
		print ('Arguments= ' , str(sys.argv))
		count = 0
		while (count < len(sys.argv)):
			print ("Argument#%d is %s" % (count, sys.argv[count]))
			count = count + 1
	else:
		# program was run without any arguments
		print ('No program arguments were provided.')



# main function
def main():
	show_arguments()
	print('Hello, World!')


# call main function
main()