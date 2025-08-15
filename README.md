# Breaking the Sorting Barrier for Directed Single-Source Shortest Paths

Dijkstra's Dethroned After 41 Years: The Breakthrough Algorithm
Yes, this is real! Researchers from Tsinghua University have achieved the first major breakthrough in shortest-path algorithms since 1984, breaking Dijkstra's "sorting barrier" with a new deterministic O(m log^(2/3) n) time algorithm. This landmark achievement won the Best Paper Award at STOC 2025.

The Breakthrough Explained
What Was the "Sorting Barrier"?
For 41 years, computer scientists believed that any shortest-path algorithm faster than Dijkstra's O(m + n log n) was impossible because finding shortest paths seemed to require sorting vertices by distance—and sorting fundamentally takes O(n log n) time. This became known as the "sorting barrier."

Key Innovation: 

No More Full Sorting: The new algorithm breaks this barrier by avoiding complete sorting of vertices. Instead, it:

- Uses recursive clustering to group neighboring frontier vertices
- Combines Dijkstra's and Bellman-Ford approaches intelligently
- Processes vertices in batches rather than individually
- Maintains partial ordering instead of complete sorting
