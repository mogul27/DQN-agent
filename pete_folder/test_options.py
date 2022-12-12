import unittest

from dqn_utils import Options


class OptionsTest(unittest.TestCase):

    def test_get_option_value(self):
        options = Options({'one': 1, 'two': 2})

        self.assertEqual(2, options.get('two'))

        # unknown options default to None
        self.assertIsNone(options.get('three'))

    def test_set_default_value(self):
        options = Options({'one': 1, 'two': 2})

        # add a default value for an option that doesn't exist
        self.assertIsNone(options.get('ok'))
        options.default('new', 'ok')
        self.assertEqual('ok', options.get('new'))

        # check default value doesn't override existing value
        self.assertEqual(2, options.get('two'))
        options.default('two', 34)
        self.assertEqual(2, options.get('two'))

    def test_create_from_options(self):
        options_1 = Options({'one': 1, 'two': 2})
        options_2 = Options(options_1)

        self.assertEqual(2, options_2.get('two'))

        # make sure it's a copy
        options_1.set('three', 3)
        self.assertEqual(2, options_1.get('two'))
        # Should not have altered options_2
        self.assertIsNone(options_2.get('three'))


if __name__ == '__main__':
    unittest.main()
