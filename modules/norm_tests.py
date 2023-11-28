import tests

# tests.test_linear()
# tests.test_bn()
# tests.test_dropout()
# tests.test_activations()
# tests.test_sequential()

# tests.test_criterions()
# tests.test_optimizers()

# tests.test_dataloader()


import numpy as np
import modules as mm
from tqdm.notebook import tqdm
np.random.seed(42)
X_train = np.random.randn(2048, 8)
X_test = np.random.randn(512, 8)
y_train = np.sin(X_train).sum(axis=1, keepdims=True)
y_test = np.sin(X_test).sum(axis=1, keepdims=True)

train_loader = mm.DataLoader(X_train, y_train, batch_size=64, shuffle=True)
test_loader = mm.DataLoader(X_test, y_test, batch_size=64, shuffle=False)

model = mm.Sequential(
    mm.Linear(8, 32),
    mm.BatchNormalization(32),
    mm.ReLU(),
    mm.Linear(32, 64),
    mm.Dropout(0.25),
    mm.Sigmoid(),
    mm.Linear(64, 1)
)
optimizer = mm.Adam(model, lr=1e-2)
criterion = mm.MSELoss()
num_epochs = 100
pbar = tqdm(range(1, num_epochs + 1))

for epoch in pbar:
    train_loss, test_loss = 0.0, 0.0

    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        model.backward(X_batch, criterion.backward(predictions, y_batch))
        optimizer.step()

        train_loss += loss * X_batch.shape[0]

    model.eval()
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        test_loss += loss * X_batch.shape[0]

    train_loss /= train_loader.num_samples()
    test_loss /= test_loader.num_samples()
    print({'train loss': train_loss, 'test loss': test_loss})
    # pbar.set_postfix({'train loss': train_loss, 'test loss': test_loss})