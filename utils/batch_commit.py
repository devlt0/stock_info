import os
import subprocess
import sys

from math import ceil

def git_add_commit(directory="/shards/", batch_size=100, commit_message="", _dry_run=True, _only_commit_mod_or_new=True):
    os.chdir(directory)

    files = []
    if _only_commit_mod_or_new:
        # Get the list of modified or newly added files (exclude deleted files)
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)

        for line in result.stdout.splitlines():
            status, filename = line.split(maxsplit=1)
            if status in ['M', 'A', '?', '??'] and filename.endswith('.db'):  # M for modified, A for added, ? ?? for untracked
                #files.append( os.path.join( directory, os.path.basename(filename) ) )
                files.append( os.path.basename(filename) )
        # compensate for git status --porcelain returns seemingly all modded files not just those in current dir with added filter
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f) and f in files)]
    else:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if not files:
        outmsg = f"No modified or new files to commit in the specified directory {directory}." \
            if _only_commit_mod_or_new else f"No files to commit in the specified directory {directory}."
        print(outmsg)
        return

    num_files = len(files)
    num_batches = ceil( num_files / batch_size )
    batch_iter = 0

    for i in range(0, num_files, batch_size):
        batch = files[i:i + batch_size]

        if _dry_run:
            print(f"Current batch {batch_iter+1} of {num_batches} includes; \n {batch}")
        else:
            # Staging
            subprocess.run(["git", "add"] + batch, check=True)

            # Commit
            cur_batch_suffix = f"\n\n Batch {batch_iter + 1} of {num_batches}"
            cur_commit_msg = commit_message + cur_batch_suffix
            subprocess.run(["git", "commit", "-m", cur_commit_msg], check=True)

            # Push
            print(f"Pushing commit for batch {batch_iter + 1} of {num_batches}...")
            subprocess.run(["git", "push"], check=True)

        batch_iter += 1

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <directory> <batch_size> <commit_message>")
        sys.exit(1)

    directory = sys.argv[1]
    batch_size = int(sys.argv[2])
    commit_message = sys.argv[3]

    # Ensure directory exists
    if not os.path.isdir(directory):
        print(f"The specified directory '{directory}' does not exist.")
        sys.exit(1)

    # Run the add and commit function
    git_add_commit(directory, batch_size, commit_message)

if __name__ == "__main__":
    #main()
    nasdaq_dir = r"C:\Users\human-c137\Documents\GitHub\stock_info\shards\nasdaq"
    nyse_dir = r"C:\Users\human-c137\Documents\GitHub\stock_info\shards\nyse"
    # ToDo make batch size dynamic based upon desired push size, def to 1gb for github
    #  batch size = floor(1gb/avg file size)
    git_add_commit(directory=nyse_dir, commit_message="nyse update mass update for converting column types (new files ~1/2 the size)",
                   batch_size=100,  _only_commit_mod_or_new=True, _dry_run=False)
    git_add_commit(directory=nasdaq_dir, commit_message="nasdaq update mass update for converting column types(new files ~1/2 the size)",
                    batch_size=100,  _only_commit_mod_or_new=True, _dry_run=False)


