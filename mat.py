from abc import ABC, abstractmethod

class SymmetricMatrix[T]:
    size: int
    data: list[T]

    def __init__(self, n: int, default: T):
        self.size = n
        self.data = [default] * (n*(n+1)//2)

    @classmethod
    def index(cls, index: tuple[T, T]):
        i, j = index
        if j > i:
            i, j = j, i
        return i*(i+1)//2 + j

    def __str__(self):
        s = ""
        for i in range(self.size):
            for j in range(self.size):
                s += f"{self.data[self.index((i, j))]} "
            s += "\n"
        return s

    def __getitem__(self, index):
        if type(index) is tuple:
            i = self.index(index)
            return self.data[i]
        return [self.data[self.index((index, j))] for j in range(self.size)]

    def __setitem__(self, index, item: T):
        if type(index) is tuple:
            i = self.index(index)
            self.data[i] = item
        else:
            for j in range(self.size):
                self.data[self.index((index, j))] = item[j]

class Matrix[T](ABC):
    size: int

    @abstractmethod
    def get(self, i: int, j: int) -> T:
        pass

    @abstractmethod
    def set(self, i: int, j: int, v: T):
        pass

    def setrow(self, i: int, v: list[T]):
        for j in range(self.size):
            self.set(i, j, v[j])

    def __getitem__(self, index: int | tuple[int, int]) -> MatrixView[T] | T:
        if type(index) is int:
            return MatrixView(self, (index, index), (0, self.size-1))
        elif type(index) is tuple:
            i, j = index
            return self.get(i, j)
        raise TypeError

    def __setitem__(self, index: int | tuple[int, int], v: list[T] | T):
        if type(index) is int:
            self.setrow(index, v)
        elif type(index) is tuple:
            i, j = index
            self.set(i, j, v)

    
class MatrixView[T]:
    m: Matrix[T]
    si: int
    ei: int
    sj: int
    ej: int

    def __init__(self, m: Matrix[T], i: tuple[int, int], j: tuple[int, int]):
        self.m = m
        self.si, self.ei = i
        self.sj, self.ej = j

    def __getitem__(self, index: int | tuple[int, int]) -> MatrixView[T] | T:
        if type(index) is int:
            return MatrixView(self.m, (self.si + index, self.si + index), (self.sj, self.ej))
        elif type(index) is tuple:
            i, j = index
            return self.m.get(self.si + i, self.sj + j)
        raise TypeError

    def __setitem__(self, index: int | tuple[int, int], v: list[T] | T):
        if type(index) is int and isinstance(v, list):
            for j in range(self.sj, self.ej):
                self.m.set(index, j, v[j - self.sj])
        elif type(index) is tuple and v is T:
            i, j = index
            self.m.set(i, j, v)

def slice_to_range(s: slice, len: int):
    start, stop, step = s.indices(len)
    return range(start, stop, step)

def func(s: slice[int, int]):
    return 1

if __name__ == "__main__":
    m = SymmetricMatrix(5, 0)
    m[1, 2] = 1
    print(m)
    print(m[1])
    m[3] = [1,2,3,4,5]
    m[0][0] = 1
    print(m)

    func(slice(1, 2, 2))
