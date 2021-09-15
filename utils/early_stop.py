import logging
logger = logging.getLogger("Ealiy Stop")

class EarlyStop():
    BEST = 1
    CONTINUE = 2
    STOP = -1

    def __init__(self,max_retry):
        self.best_value = -9999999
        self.retry_counter = 0
        self.max_retry= max_retry

    def evaluate(self,value):

        if value> self.best_value:
            logger.debug("[早停] 新值%.4f>旧值%.4f，记录最好的，继续训练",value,self.best_value)
            # 所有的都重置
            self.retry_counter = 0
            self.best_value = value
            return EarlyStop.BEST

        # 甭管怎样，先把计数器++
        self.retry_counter+=1
        logger.debug("[早停] 新值%f<旧值%f,早停计数器:%d", value, self.best_value,self.retry_counter)

        # 如果还没有到达最大尝试次数，那就继续
        if self.retry_counter < self.max_retry:
            logger.debug("[早停] 早停计数器%d未达到最大尝试次数%d，继续训练",self.retry_counter,self.max_retry)
            return EarlyStop.CONTINUE

        logger.debug("[早停] 早停计数器%d都已经达到最大，退出训练", self.retry_counter)
        # 如果到达最大尝试次数，并且也到达了最大decay次数
        return EarlyStop.STOP

    def decide(self, value, saver, *args):
        decision = self.evaluate(value)

        if decision == EarlyStop.CONTINUE:
            logger.info("新Value值比最好的要小，继续训练...")
            return False

        if decision == EarlyStop.BEST:
            logger.info("新Value值[%f]大于过去最好的Value值，早停计数器重置，并保存模型", value)
            saver(*args)
            return False

        if decision == EarlyStop.STOP:
            logger.warning("超过早停最大次数，也尝试了多次学习率Decay，无法在提高：第%d次，训练提前结束", step)
            return True

        logger.error("无法识别的EarlyStop结果：%r", decision)
        return True
