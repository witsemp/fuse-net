from Utils import list_mean
import torch


class Train:
    def __init__(self,
                 device,
                 model,
                 criterion,
                 epochs,
                 learning_rate,
                 save_path,
                 training_dataset,
                 training_loader,
                 batch_size,
                 with_validation=False,
                 valid_dataset=None,
                 valid_loader=None):

        self.device = device
        self.model = model
        self.criterion = criterion
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.training_dataset = training_dataset
        self.training_loader = training_loader
        self.valid_dataset = valid_dataset
        self.valid_loader = valid_loader

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        for epoch in range(self.epochs):
            training_losses = []
            validation_losses = []
            self.model.train()
            for i, (input_rgb, input_depth, target_depth) in enumerate(self.training_loader):
                print(f'Epoch {epoch}, batch number: {i}/{len(self.training_dataset.inputs) / batch_size}')
                self.optimizer.zero_grad()
                input_rgb, input_depth, target_depth = input_rgb.to(self.device), \
                                                       input_depth.to(self.device), \
                                                       target_depth.to(self.device)
                output_depth = model(input_rgb, input_depth)
                print(output_depth.shape)
                loss = self.criterion(output_depth, target_depth)
                loss.backward()
                self.optimizer.step()
                training_losses.append(loss.item())
            print(f'Average training loss:, {list_mean(training_losses)}')

            if with_validation:
                self.model.eval()
                with torch.no_grad():
                    for j, (input_rgb_valid, input_depth_valid, target_depth_valid) in enumerate(self.valid_loader):
                        input_rgb_valid, input_depth_valid, target_depth_valid = input_rgb_valid.to(self.device), \
                                                                                 input_depth_valid.to(self.device), \
                                                                                 target_depth_valid.to(self.device)
                        output_depth_valid = self.model(input_rgb_valid, input_depth_valid)
                        val_loss = self.criterion(output_depth_valid, target_depth_valid)
                        validation_losses.append(val_loss.item())
                        print(f'Average validation loss:, {list_mean(validation_losses)}')
        torch.save(self.model, self.save_path)
