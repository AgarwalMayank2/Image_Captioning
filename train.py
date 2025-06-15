import pre_processing
from sklearn.model_selection import train_test_split
import torch
import resnet_encoder
import transformer_decoder
import pandas as pd

num_epochs = 30
batch_size = 32
learning_rate = 0.00001

def trainer(processed_data, device):
    resnet_dataloader = transformer_decoder.get_resnet_dataloader(processed_data.data, "EncodedImageResnet.pkl", 32)

    decoderModel = transformer_decoder.decoderTransformer(16, 4, processed_data.vocab_size, 512, processed_data.max_seq_length, device).to(device)
    optimizer = torch.optim.Adam(decoderModel.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience = 2, verbose = True)
    criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
    min_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0
        total_words = 0

        for caption_seq, target_seq, image_tensor in resnet_dataloader:

            optimizer.zero_grad()

            image_tensor = image_tensor.squeeze(1).to(device)
            caption_seq = caption_seq.long().to(device)
            target_seq = target_seq.long().to(device)

            output, padding_mask = decoderModel.forward(image_tensor, caption_seq)
            output = output.permute(1, 2, 0)

            loss = criterion(output, target_seq)
            masked_loss = torch.mul(loss, padding_mask)
            batch_loss = torch.sum(masked_loss)/torch.sum(padding_mask)

            batch_loss.backward()
            optimizer.step()

            epoch_loss += torch.sum(masked_loss).detach().item()
            total_words += torch.sum(padding_mask)

        epoch_loss = epoch_loss / total_words

        scheduler.step(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        if epoch_loss < min_loss:
            torch.save(decoderModel, './BestModel')
            min_loss = epoch_loss