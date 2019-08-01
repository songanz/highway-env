import os
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
    old_string = "/s/szhan117/highway-env"
    new_string = "/home/songanz/Documents/Git_repo/highway-env"
    for file_in in os.listdir(cwd + '/scripts/bash_for_ford_hpc/'):
        with open(cwd + '/scripts/bash_for_ford_hpc/' + file_in, "rt") as fin:
            file_out = os.path.abspath(cwd + '/scripts/bash_home/' + file_in)
            with open(file_out, "wt") as fout:
                for line in fin:
                    fout.write(line.replace(old_string, new_string))