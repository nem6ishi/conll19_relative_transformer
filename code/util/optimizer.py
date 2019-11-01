class OptimizerWrapper:
  def __init__(self, optimizer, trainer=None, model_dim=None, warmup_step=None):
    self.optimizer = optimizer
    self.trainer = trainer # just to get step
    self.model_dim = model_dim
    self.warmup_step = warmup_step
    self.use_warmup = True if trainer!=None and model_dim!=None and warmup_step!=None else False


  def step(self):
    if self.use_warmup:
      step = self.trainer.step
      learning_rate = (self.model_dim**(-0.5) * min(step**(-0.5), step*self.warmup_step**(-1.5)))
      self.optimizer.param_groups[0]['lr'] = learning_rate

    self.optimizer.step()
