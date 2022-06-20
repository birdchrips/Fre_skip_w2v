from ignite.engine import Events, create_supervised_trainer
from ignite.handlers import EarlyStopping
import matplotlib.pyplot as plt


class SkipGramTrainer():
    def __init__(self, model, optimizer, loss_func, train_loader, save_file):

        self.trainer = create_supervised_trainer(model, optimizer, loss_func)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.log_training_loss)
        self.train_loader = train_loader
        self.train_loss = []
        self.save_file = save_file
        
    def run(self, max_epoch):
        self.trainer.run(self.train_loader, max_epochs=max_epoch)

    def score_function(self, engine):
        loss = engine.state.metrics['loss']
        return loss

    def log_training_loss(self, engine):
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration

        self.train_loss.append(engine.state.output)
        if i % 50 == 0:
            print(f"Epoch {e}/{n} : {i} - batch loss: {engine.state.output:.4f}")

    def print_loss_graph(self):
        plt.plot(self.train_loss, label='train loss')
        plt.legend()
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.title("loss value graph", fontsize=14)
        plt.show()


