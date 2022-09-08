import unittest

from srl.rl.functions.common import float_category_decode, float_category_encode


class Test(unittest.TestCase):
    def test_category(self):
        with self.subTest(("plus",)):
            cat = float_category_encode(2.4, -5, 5)
            self.assertAlmostEqual(cat[7], 0.6)
            self.assertAlmostEqual(cat[8], 0.4)

            val = float_category_decode(cat, -5, 5)
            self.assertAlmostEqual(val, 2.4)

        with self.subTest(("minus",)):
            cat = float_category_encode(-2.6, -5, 5)
            self.assertAlmostEqual(cat[2], 0.6)
            self.assertAlmostEqual(cat[3], 0.4)

            val = float_category_decode(cat, -5, 5)
            self.assertAlmostEqual(val, -2.6)

        with self.subTest(("out range(plus)",)):
            cat = float_category_encode(7, -2, 2)
            self.assertAlmostEqual(cat[3], 0.0)
            self.assertAlmostEqual(cat[4], 1.0)

            val = float_category_decode(cat, -2, 2)
            self.assertAlmostEqual(val, 2)

        with self.subTest(("out range(minus)",)):
            cat = float_category_encode(-7, -2, 2)
            self.assertAlmostEqual(cat[0], 1.0)
            self.assertAlmostEqual(cat[1], 0.0)

            val = float_category_decode(cat, -2, 2)
            self.assertAlmostEqual(val, -2)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_category", verbosity=2)
