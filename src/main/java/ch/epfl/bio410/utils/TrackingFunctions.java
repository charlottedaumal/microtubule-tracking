package ch.epfl.bio410.utils;


import ch.epfl.bio410.graph.DirectionVector;
import ch.epfl.bio410.graph.PartitionedGraph;
import ch.epfl.bio410.graph.Spot;


public class TrackingFunctions {

    /**
     * This method computes the direction vector of a given Spot by comparing it to a spot
     * in the previous time frame within a specified search window.
     * The function searches for the closest spot (based on Euclidean distance) in the
     * previous frame that lies within a square window of side length dimension,
     * centered around the current spot. If no such spot is found, the current spot's
     * position is reused as a fallback (resulting in a zero vector).
     *
     * @param spot The current Spot whose movement direction is to be estimated.
     * @param frames A PartitionedGraph representing the full set of tracked spots across all time frames.
     * @param dimension The size (in pixels) of the square search window used to look for the closest spot
     * in the previous frame.
     * @return A DirectionVector representing the estimated direction (dx, dy) from the closest
     * spot in the previous frame to the current spot. If no such spot is found, a zero vector is returned.
     */
    public static DirectionVector findDirection(Spot spot, PartitionedGraph frames, int dimension) {
        Spot previousSpot = null;
        int halfDim = dimension / 2;
        int tPrev = Math.max(spot.t - 1,1);
        double minDist = Double.MAX_VALUE;

        for (Spot other : frames.get(tPrev)) {
            if (other.t == tPrev) {
                double dx = other.x - spot.x;
                double dy = other.y - spot.y;
                double dist = dx * dx + dy * dy;

                if (dist < minDist && Math.abs(dx) <= halfDim && Math.abs(dy) <= halfDim) {
                    minDist = dist;
                    previousSpot = other;
                }
            }
        }

        if (previousSpot == null) {
            previousSpot = new Spot(spot.x, spot.y, tPrev, spot.value); // fallback
        }

        return new DirectionVector(spot.x - previousSpot.x, spot.y - previousSpot.y, spot);
    }
}

