import subprocess, re, os, datetime, shutil

bleu_path = os.path.dirname(__file__)+"/multi-bleu.perl"



def remove_sp_marker(string):
  string = string.replace(' ', '')
  string = string.replace('▁', ' ')
  string = string.strip()
  return string


def apply_sp(file_path):
  text = ""
  with open(file_path, "r") as file:
    for line in file:
      line = line.strip()
      line = remove_sp_marker(line) + "\n"
      text += line
  with open(file_path, "w") as file:
    file.write(text)


def apply_kytea(file_path):
  text = ""
  with open(file_path, "r") as file:
    for line in file:
      line = line.strip() + "\n"
      line = line.replace(" ", "")
      text += line
  with open(file_path, "w") as file:
    file.write(text)

  with open(file_path, "r") as read:
    kytea_cmd = ["kytea", "-out", "tok"]
    kytea_out = subprocess.check_output(kytea_cmd, stdin=read, stderr=subprocess.STDOUT)
    kytea_out = kytea_out.decode("utf-8")
  with open(file_path, "w") as file:
    file.write(kytea_out)


def calc_bleu(reference_file_path, prediction_file_path, use_sp=False, use_kytea=False, save_dir="./", add_file_name=""):

  if add_file_name:
    reference_file = save_dir + "/reference_{}.txt".format(add_file_name)
    prediction_file = save_dir + "/prediction_{}.txt".format(add_file_name)
  else:
    reference_file = save_dir + "/reference.txt"
    prediction_file = save_dir + "/prediction.txt"

  shutil.copyfile(reference_file_path, reference_file)
  shutil.copyfile(prediction_file_path, prediction_file)

  if use_sp:
    apply_sp(reference_file)
    apply_sp(prediction_file)

  if use_kytea:
    apply_kytea(reference_file)
    apply_kytea(prediction_file)

  bleu_score = 0.0
  for i in range(5): # subprocess sometimes fails
    try:
      with open(prediction_file, "r") as read_pred:
        bleu_cmd = ["perl", bleu_path] + [reference_file]
        bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT) # multi-bleu sometimes raise error
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        bleu_score = float(bleu_score)
      error = False
      break
    except:
      error = True

  return bleu_score, error
