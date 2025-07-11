# src/train.py
import torch

def train_with_adam_then_lbfgs(net, loss_function, generate_collocation_points, n_adam_epochs=1000, n_colloc=1000, n_ib=500):
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(n_adam_epochs):
        x, t = generate_collocation_points(n_colloc)
        physics_loss, ic_loss, bc_loss = loss_function(x, t, n_ib)
        loss = 0.1 * physics_loss + ic_loss + bc_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"[Adam] Epoch {epoch}, Loss: {loss.item():.4e}")

    def closure():
        optimizer_lbfgs.zero_grad()
        x, t = generate_collocation_points(n_colloc)
        loss = 0.1 * physics_loss + ic_loss + bc_loss
        loss.backward()
        return loss

    optimizer_lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=15000, line_search_fn="strong_wolfe")
    optimizer_lbfgs.step(closure)
