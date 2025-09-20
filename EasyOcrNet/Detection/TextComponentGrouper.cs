using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyOcrNet.Detection;

internal static class TextComponentGrouper
{
    public static List<SKRect> GroupIntoLines(IReadOnlyCollection<SKRect> components)
    {
        if (components.Count == 0)
        {
            return new List<SKRect>();
        }

        var ordered = components as List<SKRect> ?? components.ToList();
        ordered.Sort(static (a, b) =>
        {
            int topCompare = a.Top.CompareTo(b.Top);
            return topCompare != 0 ? topCompare : a.Left.CompareTo(b.Left);
        });

        float totalHeight = 0f;
        for (int i = 0; i < ordered.Count; i++)
        {
            totalHeight += ordered[i].Height;
        }

        float averageHeight = totalHeight / ordered.Count;
        float lineThreshold = Math.Max(5f, averageHeight * 0.5f);

        var lines = new List<LineCluster>(ordered.Count);
        foreach (var box in ordered)
        {
            float centerY = (box.Top + box.Bottom) * 0.5f;
            LineCluster? candidate = null;

            for (int i = 0; i < lines.Count; i++)
            {
                var cluster = lines[i];
                if (Math.Abs(centerY - cluster.Center) <= lineThreshold)
                {
                    candidate = cluster;
                    break;
                }
            }

            if (candidate is null)
            {
                candidate = new LineCluster(box);
                lines.Add(candidate);
            }
            else
            {
                candidate.Add(box);
            }
        }

        lines.Sort(static (a, b) => a.Center.CompareTo(b.Center));

        var results = new List<SKRect>(lines.Count);
        foreach (var cluster in lines)
        {
            results.Add(cluster.Combine());
        }

        return results;
    }

    private sealed class LineCluster
    {
        private readonly List<SKRect> _boxes = new();
        private float _center;

        public LineCluster(SKRect initial)
        {
            _boxes.Add(initial);
            _center = (initial.Top + initial.Bottom) * 0.5f;
        }

        public float Center => _center;

        public void Add(SKRect box)
        {
            _boxes.Add(box);
            float boxCenter = (box.Top + box.Bottom) * 0.5f;
            _center = (_center * (_boxes.Count - 1) + boxCenter) / _boxes.Count;
        }

        public SKRect Combine()
        {
            float left = float.PositiveInfinity;
            float top = float.PositiveInfinity;
            float right = float.NegativeInfinity;
            float bottom = float.NegativeInfinity;

            for (int i = 0; i < _boxes.Count; i++)
            {
                var box = _boxes[i];
                if (box.Left < left) left = box.Left;
                if (box.Top < top) top = box.Top;
                if (box.Right > right) right = box.Right;
                if (box.Bottom > bottom) bottom = box.Bottom;
            }

            return new SKRect(left, top, right, bottom);
        }
    }
}
