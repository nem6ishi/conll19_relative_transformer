import torch, os, re, glob, json, math, tensorboardX, datetime
from logging import getLogger
logger = getLogger(__name__)

from code.util import language, corpus, model_check, optimizer, evaluation
from code.model import seq2seq, transformer, rnn_transformer, rel_transformer, rnn_rel_transformer, posemb_transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Processor:
  def __init__(self, setting, mode):

    self.setting = setting

    self.src_lang = None
    self.tgt_lang = None
    self.train_corpus = None
    self.dev_corpus = None
    self.test_corpus = None

    self.step = 0
    self.criterion = None
    self.model = None
    self.opt_wrapper = None
    self.best_step = 0
    self.best_bleu = 0.0

    self.mode = mode

    logger.info(json.dumps(self.setting, indent = 2))
    torch.set_num_threads(4)



  def run(self):
    self.check_previous_training()
    self.load_data()
    self.prepare_model()

    if self.mode=="train":
      self.train_process()
    elif self.mode=="test":
      self.test_process()
    else:
      raise



  def check_previous_training(self):
    logger.info("Checking previous training")
    # check model dir
    model_dir = self.setting["paths"]["model_directory"]
    if not os.path.isdir(model_dir):
      os.makedirs(model_dir)

    # check previous training
    setting_path = model_dir+'/setting.json'
    if not os.path.isfile(setting_path):
      with open(setting_path, 'w') as f:
        json.dump(self.setting, f, indent=2) # save new setting
    else:
      with open(setting_path, 'r') as f:
        saved_setting = json.load(f)
      if self.mode=="train" and self.setting != saved_setting:
        raise ValueError("Setting has been changed")

    # find last checkpoint
    saved_models = glob.glob(model_dir+'/models/step_*')
    if len(saved_models) > 0:
      saved_steps = []
      for each in saved_models:
        step = int(re.search(r"[0-9]+\Z", each).group(0))
        saved_steps.append(step)
      for each in sorted(saved_steps, reverse=True):
        saved_model_path = model_dir + '/models/step_' + str(each) + "/model.pt"
        if os.path.isfile(saved_model_path):
          self.step = each
          break



  def load_data(self):
    logger.info("Loading data")
    self.src_lang = language.Lang(self.setting["paths"]["src_vocab"])
    if self.setting["paths"]["src_vocab"] == self.setting["paths"]["tgt_vocab"]:
      self.tgt_lang = self.src_lang
    else:
      self.tgt_lang = language.Lang(self.setting["paths"]["tgt_vocab"])
    assert self.tgt_lang.vocab2index["PADDING"] == self.src_lang.vocab2index["PADDING"]

    if self.mode=="train":
      if "src_train" in self.setting["paths"] and "tgt_train" in self.setting["paths"]:
        logger.info("Loading train corpus")
        self.train_corpus = corpus.ParallelCorpus(self.src_lang,
                                                       self.tgt_lang,
                                                       self.setting["paths"]["src_train"],
                                                       self.setting["paths"]["tgt_train"],
                                                       max_length=self.setting["train_vars"]["train_max_length"],
                                                       num_imort_line=self.setting["train_vars"]["train_num_import_line"])
        self.train_corpus.import_file()
        logger.info("train corpus size: {}".format(self.train_corpus.corpus_size))

      if "src_dev" in self.setting["paths"] and "tgt_dev" in self.setting["paths"]:
        logger.info("Loading dev corpus")
        self.dev_corpus = corpus.ParallelCorpus(self.src_lang,
                                                     self.tgt_lang,
                                                     self.setting["paths"]["src_dev"],
                                                     self.setting["paths"]["tgt_dev"])
        self.dev_corpus.import_file()
        logger.info("dev corpus size: {}".format(self.dev_corpus.corpus_size))

    if self.mode=="test":
      if "src_test" in self.setting["paths"]:
        logger.info("Loading test corpus")
        self.test_corpus = corpus.ParallelCorpus(self.src_lang,
                                                      self.tgt_lang,
                                                      self.setting["paths"]["src_test"],
                                                      self.setting["paths"]["tgt_test"])
        self.test_corpus.import_file()
        logger.info("test corpus size: {}".format(self.test_corpus.corpus_size))



  def select_model(self):
    if self.setting["train_vars"]["model_type"] == "seq2seq":
      self.model = seq2seq.Seq2SeqModel(self.setting, self.src_lang, self.tgt_lang)

    elif self.setting["train_vars"]["model_type"] == "transformer":
      self.model = transformer.TransformerModel(self.setting, self.src_lang, self.tgt_lang)

    elif self.setting["train_vars"]["model_type"] == "rnn_transformer":
      self.model = rnn_transformer.RNNTransformerModel(self.setting, self.src_lang, self.tgt_lang)

    elif self.setting["train_vars"]["model_type"] == "rel_transformer":
      self.model = rel_transformer.RelTransformerModel(self.setting, self.src_lang, self.tgt_lang)

    elif self.setting["train_vars"]["model_type"] == "rnn_rel_transformer":
      self.model = rnn_rel_transformer.RNNRelTransformerModel(self.setting, self.src_lang, self.tgt_lang)

    elif self.setting["train_vars"]["model_type"] == "posemb_transformer":
      self.model = posemb_transformer.TransformerModel(self.setting, self.src_lang, self.tgt_lang)

    else:
      raise

    self.model.to(device)



  def prepare_model(self):
    logger.info("Preparing model")
    self.select_model()
    logger.info("Model info:\n{}".format(self.model))
    logger.info("Model on: {}".format(device))
    logger.info("Model # of param: {}".format(model_check.count_num_params(self.model)))
    if "encoder" in self.model.__dict__["_modules"]:
      logger.info("Encoder # of param: {}".format(model_check.count_num_params(self.model.encoder)))
    if "decoder" in self.model.__dict__["_modules"]:
      logger.info("Decoder # of param: {}".format(model_check.count_num_params(self.model.decoder)))
    if "generator" in self.model.__dict__["_modules"]:
      logger.info("Generator # of param: {}".format(model_check.count_num_params(self.model.generator)))

    if self.mode == "train":
      self.opt_wrapper = optimizer.OptimizerWrapper(torch.optim.Adam(self.model.parameters(),
                                                                     lr=self.setting["train_vars"]["learning_rate"]),
                                                                     trainer = self,
                                                                     model_dim = self.setting["encoder_vars"]["model_dim"],
                                                                     warmup_step = 4000)
      self.criterion = torch.nn.NLLLoss(ignore_index=self.tgt_lang.vocab2index["PADDING"])

    # load previous model
    if self.step == 0:
      logger.info("New model")
    else:
      logger.info("Load model")
      load_model_path = self.setting["paths"]["model_directory"] + '/models/step_' + str(self.step) + "/model.pt"
      load_model = torch.load(load_model_path)
      if self.mode == "train":
        logger.info("Restart training")
        self.model.load_state_dict(load_model['model_state_dict'])
        self.opt_wrapper.optimizer.load_state_dict(load_model['optimizer_state_dict'])
        self.best_step = load_model['best_step']
        self.best_bleu = load_model['best_bleu']
      if self.mode == "test":
        if "step" in self.setting["pred_vars"]:
          self.step = self.setting["pred_vars"]["step"]
        else:
          self.step = load_model['best_step']
        load_model_path = self.setting["paths"]["model_directory"] + '/models/step_' + str(self.step) + "/model.pt"
        load_model = torch.load(load_model_path)
        self.model.load_state_dict(load_model['model_state_dict'])

    logger.info("Model ready. Current step: {}".format(self.step))



  def tensorboard_log(self, step, tmp_bleu, best_bleu, ave_loss=False, tmp_loss=False, ave_norm=False, tmp_norm=False):
    log_dir=self.setting["paths"]["model_directory"] + '/tb_logs/'
    writer = tensorboardX.SummaryWriter(log_dir=log_dir)
    writer.add_scalar('bleu_score/tmp', tmp_bleu, step)
    writer.add_scalar('bleu_score/best', best_bleu, step)
    if ave_loss:
      writer.add_scalar('loss/average', ave_loss, step)
    if tmp_loss:
      writer.add_scalar('loss/tmp', tmp_loss, step)
    if ave_loss:
      writer.add_scalar('norm/average', ave_norm, step)
    if ave_loss:
      writer.add_scalar('norm/tmp', tmp_norm, step)
    writer.close()



  def train_process(self):
    logger.info("Start training")
    if self.step == 0:
      self.tensorboard_log(self.step, 0.0, 0.0)
    num_cat_sent = self.setting["train_vars"]["num_cat_sent"] if "num_cat_sent" in self.setting["train_vars"] else 1
    batch = corpus.ParallelBatch(self.train_corpus, self.setting["train_vars"]["batch_size"], num_cat_sent=num_cat_sent)
    reset_flag = True


    while self.step < self.setting["train_vars"]["step"]:
      if reset_flag:
        average_loss, average_norm = 0.0, 0.0
        reset_flag=False
        logger.info("Train model for {0} steps".format(self.setting["train_vars"]["check_steps"]))

      self.step += 1
      batch.generate_random_indexes()
      batch.generate_batch()
      loss, norm = self.train_one_step(batch)
      average_loss += loss
      average_norm += norm

      if self.step%(self.setting["train_vars"]["check_steps"]//10)==0:
        logger.info("{} steps done.\tLoss: {}".format(self.step, loss))

      if self.step%self.setting["train_vars"]["check_steps"]==0:
        # validation and save
        tmp_score = self.validation()
        self.save(tmp_score)

        # save data for tensorboard
        self.tensorboard_log(self.step,
                             tmp_score,
                             self.best_bleu,
                             ave_loss=average_loss/self.setting["train_vars"]["check_steps"],
                             tmp_loss=loss,
                             ave_norm=average_norm/self.setting["train_vars"]["check_steps"],
                             tmp_norm=norm)
        reset_flag=True



  def train_one_step(self, batch):
    self.model.train()
    self.opt_wrapper.optimizer.zero_grad()

    prob_outputs = self.model.translate_for_train(batch)
    loss = self.criterion(prob_outputs.contiguous().view(-1, self.tgt_lang.vocab_size),
                          batch.tgt_batch.sentences[:, 1:].contiguous().view(-1))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.setting["train_vars"]["clip_grad_norm"])
    self.opt_wrapper.step()

    return float(loss.data), norm


  def prediction(self, given_corpus, batch_size):
    logger.info("Prediction with batch size of {}".format(batch_size))
    batch = corpus.ParallelBatch(given_corpus, batch_size)

    self.model.eval()
    prediction_sents = []

    likelihood_data = []

    with torch.no_grad():
      for i in range(math.ceil(given_corpus.corpus_size / batch_size)):
        num_rest = given_corpus.corpus_size - i*batch_size
        logger.info("{} sents to go".format(num_rest))

        batch.generate_sequential_indexes(i)
        batch.generate_batch()

        word_outputs, prob_outputs = self.model.translate(batch,
                                                          self.setting["train_vars"]["prediction_max_length"])

        for each in list(word_outputs):
          sent = self.tgt_lang.indexes2sentence(each)
          prediction_sents.append(sent)

    return prediction_sents



  def validation(self):
    logger.info("Validation")
    prediction_sents = self.prediction(self.dev_corpus, batch_size=self.setting["train_vars"]["valid_batch_size"])

    logger.info("Save dev prediction")
    save_dir = self.setting["paths"]["model_directory"] + '/models/step_' + str(self.step)
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    pred_save_file = save_dir + "/raw_prediction.txt"
    with open(pred_save_file, "w") as file:
      for sent in prediction_sents:
        sent = ' '.join(sent)
        file.write(sent + "\n")

    logger.info("Calculating bleu score")
    score, error = evaluation.calc_bleu(self.dev_corpus.tgt_corpus.file_path,
                                             pred_save_file,
                                             self.setting["options"]["sentence_piece"],
                                             self.setting["options"]["use_kytea"],
                                             save_dir=save_dir)
    logger.info("Bleu score: {} @step {}".format(score, self.step))

    # save bleu_score
    bleu_file = self.setting["paths"]["model_directory"] + "/bleu_score.txt"
    line = str(self.step) + "\t" + str(score)
    if error:
       line += "\t*ERROR"
    with open(bleu_file, "a") as file:
      file.write(line + "\n")

    return score



  def save(self, score):
    # update best score & step
    prev_best_step = self.best_step
    if self.best_bleu <= score:
      self.best_step, self.best_bleu = self.step, score
    logger.info("Best bleu score: {} @step {}".format(self.best_bleu, self.best_step))

    # save current model
    logger.info("Save model")
    save_dir = self.setting["paths"]["model_directory"] + '/models/step_' + str(self.step)
    save_model_path = save_dir + "/model.pt"
    if not os.path.isdir(save_dir):
      os.mkdir(save_dir)
    elif os.path.isfile(save_model_path):
      raise ValueError("Overwriting saved model.")
    model_to_save = {'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict' : self.opt_wrapper.optimizer.state_dict(),
                     'best_step': self.best_step,
                     'best_bleu': self.best_bleu}
    torch.save(model_to_save, save_model_path)
    logger.info("Saved")

    # delete unnecessary models
    models_to_delete = []
    prev_step = self.step - self.setting["train_vars"]["check_steps"]
    if prev_step != self.best_step and prev_step != 0:
      models_to_delete.append(prev_step)
    if prev_best_step != self.best_step and prev_best_step != 0:
      models_to_delete.append(prev_best_step)
    for each in models_to_delete:
      model_path = self.setting["paths"]["model_directory"] + '/models/step_' + str(each) +"/model.pt"
      if os.path.isfile(model_path):
        logger.info("Delete model of step {}".format(each))
        os.remove(model_path)



  def test_process(self):
    logger.info("Prediction")
    if self.mode=="train":
      batch_size = self.setting["train_vars"]["valid_batch_size"]
    elif self.mode=="test":
      batch_size = self.setting["pred_vars"]["batch_size"]

    prediction_sents = self.prediction(self.test_corpus, batch_size=batch_size)

    logger.info("Save test prediction")
    save_dir = self.setting["paths"]["model_directory"] + '/test/'
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    pred_save_file = save_dir + "/raw_test_prediction_{}.txt".format(time_stamp)
    with open(pred_save_file, "w") as file:
      for sent in prediction_sents:
        sent = ' '.join(sent)
        file.write(sent + "\n")

    if self.setting["paths"]["tgt_test"]:
      logger.info("Calculating bleu score")
      score, error = evaluation.calc_bleu(self.test_corpus.tgt_corpus.file_path,
                                               pred_save_file,
                                               self.setting["options"]["sentence_piece"],
                                               self.setting["options"]["use_kytea"],
                                               save_dir=save_dir,
                                               add_file_name=time_stamp)
      logger.info("Bleu score: {} @step {}".format(score, self.step))
