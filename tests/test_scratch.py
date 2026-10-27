import numpy as np
from src.from_scratch.activations import ReLU, SoftmaxCrossEntropy
from src.from_scratch.layers import Dense
from src.from_scratch.network import NeuralNetwork
from src.from_scratch.optimizers import Adam


def test_dense_forward_backward_shapes():
    batch_size = 16
    in_dim = 3072
    out_dim = 128

    layer = Dense(in_dim, out_dim)
    x = np.random.randn(batch_size, in_dim).astype(np.float32)

    out = layer.forward(x)
    assert out.shape == (batch_size, out_dim)

    grad_out = np.random.randn(batch_size, out_dim).astype(np.float32)
    grad_in = layer.backward(grad_out)

    assert grad_in.shape == (batch_size, in_dim)
    assert layer.grad_W.shape == (in_dim, out_dim)
    assert layer.grad_b.shape == (1, out_dim)


def test_relu():
    relu = ReLU()
    x = np.array([[-2.0, 0.0, 3.0]], dtype=np.float32)
    out = relu.forward(x)
    assert np.allclose(out, [[0.0, 0.0, 3.0]])

    grad = relu.backward(np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
    assert np.allclose(grad, [[0.0, 0.0, 1.0]])


def test_softmax_cross_entropy():
    loss_fn = SoftmaxCrossEntropy()
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]], dtype=np.float32)
    y_true = np.array([0, 1], dtype=np.int64)

    loss = loss_fn.forward(logits, y_true)
    assert loss > 0.0

    grad = loss_fn.backward()
    assert grad.shape == logits.shape
    assert np.allclose(np.sum(grad, axis=-1), [0.0, 0.0], atol=1e-5)


def test_network_learns_overfit_batch():
    np.random.seed(42)
    X = np.random.randn(32, 64).astype(np.float32)
    y = np.random.randint(0, 3, size=(32,))

    net = NeuralNetwork([Dense(64, 32, init_method="he"), ReLU(), Dense(32, 3)])
    opt = Adam(lr=0.01)

    initial_loss, _ = net.evaluate(X, y)
    history = net.fit(X, y, X, y, epochs=10, batch_size=32, optimizer=opt, verbose=False)
    final_loss = history["train_loss"][-1]

    assert final_loss < initial_loss
