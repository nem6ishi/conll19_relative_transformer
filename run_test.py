import argparse, json, logging, os

from logging import getLogger, basicConfig
fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
basicConfig(level=logging.INFO, format=fmt)
logger = getLogger(__name__)

import code.processor


parser = argparse.ArgumentParser(description='A script to test NMT model.')
parser.add_argument('-d', '--directory', required=True)

parser.add_argument('--src_test', required=True)
parser.add_argument('--tgt_test', default=None, required=False)

parser.add_argument('--step')
parser.add_argument('--batch_size', default=32, required=False)
parser.add_argument('--prediction_max_length', default=200, required=False)

args = parser.parse_args()


if __name__ == '__main__':
  setting_path = args.directory+'/setting.json'
  if os.path.isfile(setting_path):
    with open(setting_path, 'r') as f:
      setting = json.load(f)
  else:
    raise

  setting["paths"]["src_test"] = args.src_test
  if args.tgt_test:
    setting["paths"]["tgt_test"] = args.tgt_test

  setting["pred_vars"] = {}
  if args.step:
    setting["pred_vars"]["step"] = int(args.step)
  setting["pred_vars"]["batch_size"] = int(args.batch_size)
  setting["pred_vars"]["prediction_max_length"] = int(args.prediction_max_length)

  predictor = code.processor.Processor(setting, mode="test")
  predictor.run()
