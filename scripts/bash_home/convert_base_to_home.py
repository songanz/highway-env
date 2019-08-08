import os
import glob
'''
Run this file with in the project folder! i.e.: highway-env
'''
def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        s = f.read()
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)


if __name__ == "__main__":
    cwd = os.getcwd()
    path = os.path.abspath(cwd + '/scripts/bash_for_ford_hpc/')
    new_path = os.path.abspath(cwd + '/scripts/bash_home/')
    old_string = "/s/szhan117/highway-env"
    new_string = "/home/songanz/Documents/Git_repo/highway-env"
    # for file_in in os.listdir(cwd + '/scripts/bash_for_ford_hpc/'):
    # r=root, d=directories, f = files
    files = [f for f in glob.glob(path + "**/**/*.sh", recursive=True)]
    for file_in in files:
        with open(file_in, "rt") as fin:
            file_in = file_in.split(path)[1]
            file_out = os.path.abspath(new_path + file_in)
            if not os.path.exists(os.path.dirname(file_out)): os.mkdir(os.path.dirname(file_out))
            with open(file_out, "wt") as fout:
                for line in fin:
                    fout.write(line.replace(old_string, new_string))