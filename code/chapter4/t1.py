def get_primes_up_to_n(n):
    """
    返回1到n之间的所有素数。

    参数:
    n (int): 要检查的数，范围必须大于等于2。

    返回:
    list: 1到n之间的所有素数。

    示例:
    >>> get_primes_up_to_n(10)
    [2, 3, 5, 7]
    """
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    return primes

print(get_primes_up_to_n(100))
