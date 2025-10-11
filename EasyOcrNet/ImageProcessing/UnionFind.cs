using System;

namespace EasyOcrNet.ImageProcessing;

/// <summary>
/// Union-Find (Disjoint Set Union) data structure with path compression and union by rank.
/// Used for efficiently managing equivalence classes in connected components labeling.
/// </summary>
internal sealed class UnionFind
{
    private readonly int[] _parent;
    private readonly int[] _rank;

    /// <summary>
    /// Initializes a new instance with the specified capacity.
    /// </summary>
    /// <param name="capacity">The number of elements.</param>
    public UnionFind(int capacity)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be positive.");

        _parent = new int[capacity];
        _rank = new int[capacity];

        // Initialize: each element is its own parent (separate set)
        for (int i = 0; i < capacity; i++)
        {
            _parent[i] = i;
            _rank[i] = 0;
        }
    }

    /// <summary>
    /// Finds the root representative of the set containing the specified element.
    /// Uses path compression for optimization.
    /// </summary>
    /// <param name="element">The element to find.</param>
    /// <returns>The root representative of the set.</returns>
    public int Find(int element)
    {
        if (element < 0 || element >= _parent.Length)
            throw new ArgumentOutOfRangeException(nameof(element));

        // Path compression: make every node on the path point directly to the root
        if (_parent[element] != element)
        {
            _parent[element] = Find(_parent[element]);
        }

        return _parent[element];
    }

    /// <summary>
    /// Unites the sets containing the two specified elements.
    /// Uses union by rank for optimization.
    /// </summary>
    /// <param name="element1">The first element.</param>
    /// <param name="element2">The second element.</param>
    public void Union(int element1, int element2)
    {
        int root1 = Find(element1);
        int root2 = Find(element2);

        if (root1 == root2)
            return; // Already in the same set

        // Union by rank: attach smaller tree under root of larger tree
        if (_rank[root1] < _rank[root2])
        {
            _parent[root1] = root2;
        }
        else if (_rank[root1] > _rank[root2])
        {
            _parent[root2] = root1;
        }
        else
        {
            _parent[root2] = root1;
            _rank[root1]++;
        }
    }

    /// <summary>
    /// Determines whether two elements are in the same set.
    /// </summary>
    /// <param name="element1">The first element.</param>
    /// <param name="element2">The second element.</param>
    /// <returns>True if both elements are in the same set; otherwise, false.</returns>
    public bool Connected(int element1, int element2)
    {
        return Find(element1) == Find(element2);
    }
}
