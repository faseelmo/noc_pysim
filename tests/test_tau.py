
from training.utils import adjusted_kendalls_tau

def test_perfect_positive_correlation():
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    tau = adjusted_kendalls_tau(x, y)
    assert tau == 1.0
test_perfect_positive_correlation()


def test_perfect_negative_correlation(): 
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    tau = adjusted_kendalls_tau(x, y)
    assert tau == -1.0  
test_perfect_negative_correlation()


def test_some_correlation_1():
    x = [1, 2, 3]
    y = [2, 1, 4]
    tau = adjusted_kendalls_tau(x, y)
    assert tau == 0.33   
test_some_correlation_1()

def test_with_thresholds_1(): 
    x = [1, 1, 4, 4, 8, 8]
    y = [1, 1, 4, 4, 8 ,8]
    tau = adjusted_kendalls_tau(x, y, t_x=1, t_y=1)
    assert tau == 1.0

def test_with_equal_rank(): 
    x = [1, 2, 3, 5, 3, 8]
    y = [1, 3, 5, 6, 5 ,7]
    tau = adjusted_kendalls_tau(x, y, t_x=0, t_y=0)
    assert tau == 1.0
test_with_equal_rank()
