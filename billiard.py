"""
Copyright © 2022 Naver Corporation. All rights reserved.

Compute billiard words, one letter at a time, using a generator.
"""


def billiard_word(frequency):
    """
    We break ties by order in the input.

    Example
    ------
    To make the first 20 letters of the balanced sequence
        0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ...,
    which repeats every 5 letters, do
        from billiard_word import billiard_word
        gen = billiard_word([2/5, 3/5])
        sequence = [next(gen) for _ in range(20)]
    """
    import heapq

    assert all(_ > 0 for _ in frequency)
    assert abs(sum(frequency) - 1.) < 1e-9

    tiny = 1e-9  # control roundoff issues for finite words
    heap = [(tiny*_, _) for _ in range(len(frequency))]
    while True:
        phase, letter = heapq.heappop(heap)
        heapq.heappush(heap, (phase + 1./frequency[letter], letter))
        yield letter
