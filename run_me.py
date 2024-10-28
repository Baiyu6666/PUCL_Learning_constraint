import sys

if __name__ == "__main__":
    file_to_run = sys.argv[1]

    # choosing the specified file
    # all files must ignore the first argument passed via command line.
    file_to_run = sys.argv[1]

    # Run specified file. All files must ignore the first argument
    # passed via command line
    if file_to_run == "pucl":
        from icrl.pucl import main
    elif file_to_run == "dscl":
        from icrl.dscl import main
    elif file_to_run == "run_policy":
        from icrl.run_policy import main
    else:
        raise ValueError("File %s not defined" % file_to_run)

    # running
    main()
