import subprocess
import os


weights_path_to_folder = "weights"
weights_fname = "some.pth"
bashCommand = f"./gdrive upload  /home/{os.path.join(weights_path_to_folder, weights_fname)}"

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
