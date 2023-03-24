from torch import nn

class DefaultColor(nn.Module):
    def __init__(self):
        super(DefaultColor, self).__init__()

        self.num50 = 50.
        self.num100 = 100.
        self.num110 = 110.

    def normalize_l(self, inLine):
        return (inLine - self.num50) / self.num100

    def unnormalize_l(self, inLine):
        return inLine * self.num100 + self.num50

    def normalize_ab(self, inAb):
        return inAb / self.num110

    def unnormalize_ab(self, inAb):
        return inAb * self.num110

