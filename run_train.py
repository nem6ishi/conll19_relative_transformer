import argparse, json, logging

from logging import getLogger, basicConfig
fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
basicConfig(level=logging.INFO, format=fmt)
logger = getLogger(__name__)

import code.processor


parser = argparse.ArgumentParser(description='A script to train NMT model.')
parser.add_argument('-s', '--setting', required=True)
parser.add_argument('-d', '--directory', required=False)

parser.add_argument('-vs', '--vocabsrc', required=False)
parser.add_argument('-vt', '--vocabtgt', required=False)
parser.add_argument('-ts', '--trainsrc', required=False)
parser.add_argument('-tt', '--traintgt', required=False)
parser.add_argument('-ds', '--devsrc', required=False)
parser.add_argument('-dt', '--devtgt', required=False)

args = parser.parse_args()


if __name__ == '__main__':
  with open(args.setting, 'r') as f:
    setting = json.load(f)

  if args.directory:
    setting["paths"]["model_directory"] = args.directory

  if args.vocabsrc:
    setting["paths"]["src_vocab"] = args.vocabsrc
  if args.vocabtgt:
    setting["paths"]["tgt_vocab"] = args.vocabtgt
  if args.trainsrc:
    setting["paths"]["src_train"] = args.trainsrc
  if args.traintgt:
    setting["paths"]["tgt_train"] = args.traintgt
  if args.devsrc:
    setting["paths"]["src_dev"] = args.devsrc
  if args.devtgt:
    setting["paths"]["tgt_dev"] = args.devtgt

  trainer = code.processor.Processor(setting, mode="train")
  trainer.run()
