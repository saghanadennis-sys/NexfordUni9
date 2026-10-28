# Unit Test

import unittest

class TestKMeans(unittest.TestCase):

    def test_cluster_count(self):
        self.assertEqual(kmeans.n_clusters, 5)

# Run test
unittest.main(argv=[''], exit=False)