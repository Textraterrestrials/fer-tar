import argparse
import data
import models
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch import cuda
import torch.nn.functional as F
import torch
import time
import json
from transformers import get_linear_schedule_with_warmup

MODEL_PATH = 'results/best_bilstm_w_gpt2_2.pt'
TEST_RESULTS_PATH = 'results/blstm_w_gpt2_test_2.json'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1 - 1e-5)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--clip', type=float, default=None)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--rnn', choices=['vanilla', 'lstm', 'gru'], default='lstm')
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--model_name', default='best_bilstm_gpt2_final')
    parser.add_argument('--data_file', default=None)
    parser.add_argument('--answers_file', default=None)
    args = parser.parse_args()

    RESULTS_DIR = 'rezultati_smoll_5000/'
    if args.model_name is not None:
        MODEL_PATH = 'ptmodels/' + args.model_name + '.pt'
        TEST_RESULTS_PATH = RESULTS_DIR + args.model_name + '.json'

    train_dataset = data.NLPDataset.from_file(
        args.data_file, args.answers_file
        # '../../Training/backtranslation_data3.csv',
        # '../../Training/backtranslation_labels.csv',
        # '../../Training/Gudi_GPT_data_final_log.csv',
        # '../../Training/Gudi_GPT_answers_final_log.csv',
        # '../../Training/GPT2_data_final_log.csv',
        # '../../Training/GPT2_answers_final_log.csv',
        # '../../Training/GPT2_data_final.csv',
        # '../../Training/GPT2_answers_final.csv',
        # '../../Training/subtaskA_data_all.csv',
        # '../../Training/subtaskA_answers_all.csv',
        # '../../smoll_data.csv',
        # '../../smoll_answers.csv',
    )
    text_vocab = train_dataset.text_vocab
    # test_dataset = data.NLPDataset.from_file('data/sst_test_raw.csv', text_vocab, None)
    val_dataset = data.NLPDataset.from_file(
        '../../Dev/subtaskA_dev_data.csv',
        '../../Dev/subtaskA_gold_answers.csv',
        train_dataset.text_vocab,
    )

    test_dataset = data.NLPDataset.from_file(
        '../../Test/subtaskA_test_data.csv',
        '../../Test/subtaskA_gold_answers.csv',
        train_dataset.text_vocab,
    )

    embedding = text_vocab.create_embedding_matrix(args.embedding_size,
                                                   path_to_embeddings='../../sst_glove_6b_300d.txt')
    model = models.LSTMModel(
        embedding,
        input_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )

    # ispis parametara
    r = 0
    gen = model.parameters()
    next(gen)
    for param in gen:
        r += torch.prod(torch.tensor(param.shape))
        print(param.shape)
        print(r)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    training_steps = args.epochs * (len(train_dataset) // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=training_steps,
        num_warmup_steps=training_steps // 8
    )

    criterion = F.binary_cross_entropy

    if cuda.is_available():
        print("Using GPU")
        model.to('cuda')

    best_val_acc = 0.0  # zapravo nije najbolji acc, nego acc od onog koji ima best_val_loss
    best_val_loss = 1e6
    epoch_end = 0
    for epoch in range(args.epochs):
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=data.pad_collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                    collate_fn=data.pad_collate_fn)

        models.train(model, train_dataloader, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                     clip=args.clip, epoch=epoch)
        metrics = models.evaluate(model, val_dataloader, criterion, epoch)
        print(f"Epoch {epoch + 1}: validation loss = {metrics['loss']} validation accuracy: {metrics['accuracy']}")
        # if configured, save the model if it is the best
        if args.save_best is True and metrics['accuracy'] > best_val_acc:
            best_val_loss = metrics['loss']
            best_val_acc = metrics['accuracy']
            epoch_end = epoch
            torch.save(model, MODEL_PATH)

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=data.pad_collate_fn)
    # metrics = models.evaluate(model, test_dataloader, criterion)
    # print()
    # print(f"Test loss = {metrics['loss']} test accuracy = {metrics['accuracy']}")
    time.sleep(1)
    cuda.empty_cache()
    time.sleep(1)

    # testing phase
    if args.save_best is True:
        model = torch.load(MODEL_PATH)
        model.to('cuda')

    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    test_metrics = models.evaluate(model, test_dataloader, criterion, 0, log_predictions=True)
    test_metrics['name'] = args.model_name
    test_metrics['epoch_end'] = epoch_end
    test_metrics['val_loss'] = best_val_loss
    test_metrics['val_acc'] = best_val_acc

    dictionary = {
        'name': args.model_name,
        'drop': model.lstm.dropout,
        'weight_decay': args.weight_decay,
        'epoch_end': epoch_end,
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'test_acc': test_metrics['accuracy'],
        'test_loss': test_metrics['loss'],
        'predictions': test_metrics['predictions'].float().flatten().tolist(),
    }

    with open(TEST_RESULTS_PATH, 'w') as out:
        out.write(json.dumps(dictionary, indent=4))
        print(f"{args.model_name} results saved in '{TEST_RESULTS_PATH}'")
