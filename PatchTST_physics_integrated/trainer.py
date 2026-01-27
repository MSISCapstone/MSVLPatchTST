model = Stage4PhysicsModel(config).to(device)


# Sanity check (run ONCE)
model.train()
x_long, x_short, y_true = next(iter(train_loader))

x_long = x_long.to(device)
x_short = x_short.to(device)
y_true = y_true.to(device)

y_pred = model(x_long, x_short)
loss = criterion(y_pred, y_true)

loss.backward()
optimizer.step()

print("Sanity check loss:", loss.item())
