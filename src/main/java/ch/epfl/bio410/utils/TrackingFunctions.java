package ch.epfl.bio410.utils;

import ch.epfl.bio410.graph.DirectionVector;
import ch.epfl.bio410.graph.PartitionedGraph;
import ch.epfl.bio410.graph.Spot;
import ch.epfl.bio410.graph.Spots;

public class TrackingFunctions {

    /**
     * This method compute the direction of a given spot based on the position of the spot on a previous frame
     *
     * @param spot the current spot we want to find the direction of
     * @param frames TODO
     * @param dimension the dimension of the window centered around the current spot we expect the previous spot to be in
     * @return DirectionVector, an object that stores the x and y direction of the current spot
     */
    public static DirectionVector findDirection(Spot spot, PartitionedGraph frames, int dimension) {
        Spot previousSpot = null;
        int halfDim = dimension / 2;
        int tPrev = Math.max(spot.t - 1,1);
        double minDist = Double.MAX_VALUE;

        for (Spot other : frames.get(tPrev)) {
            // TODO use getPArtitionOf instead of iterating over all spots
            // 	or for(Spot next : frames.get(t+1)) { // iterate over all spots of the next frame
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

