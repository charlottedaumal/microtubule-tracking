package ch.epfl.bio410.cost;

import ch.epfl.bio410.graph.DirectionVector;
import ch.epfl.bio410.graph.PartitionedGraph;
import ch.epfl.bio410.graph.Spot;
import ch.epfl.bio410.graph.Spots;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.ZProjector;
import ch.epfl.bio410.utils.TrackingFunctions;

public class DirectionCost implements AbstractDirCost {

    private double lambda = 0; // parameter for the intensity cost
    private double gamma = 0; // parameter for the direction cost
    private double kappa = 0; // parameter for the speed cost
    private double costMax = 0;

    /** normalization direction*/
    private double normDir =1;

    /** normalization distance */
    private double normDist = 1;

    /** normalization intensity */
    private double normInt = 1;


    public DirectionCost(ImagePlus imp, double costMax, double lambda, double gamma, double kappa) {
        this.lambda = lambda;
        this.gamma = gamma;
        this.kappa = kappa;
        this.costMax = costMax;
        int height = imp.getHeight();
        int width = imp.getWidth();
        this.normDir = 1;
        this.normDist = Math.sqrt(height * height + width * width);
        this.normInt = ZProjector.run(imp,"max").getStatistics().max - ZProjector.run(imp,"min").getStatistics().min;
    }

    @Override
    public double evaluate(Spot a, Spot b, PartitionedGraph frames, int dimension) {
        double maxAllowedDistance = 10;
        SimpleDistanceCost dist = new SimpleDistanceCost(costMax);

        double intensityDiff = Math.abs(a.value - b.value);
        double distance = dist.evaluate(a, b);

        DirectionVector dira = TrackingFunctions.findDirection(a, frames, dimension);
        DirectionVector dirb = TrackingFunctions.findDirection(b, frames, dimension);

        double directionSimilarity = dira.cosineSimilarity(dirb); // value between -1 and 1

        if (distance > maxAllowedDistance) return Double.POSITIVE_INFINITY;

        // Combine into a cost (lower is better)
        return this.lambda * dist.evaluate(a, b) / this.normDist +
                this.gamma * (1 - Math.max(0, directionSimilarity)) +
                (1 - this.lambda - this.gamma)*Math.abs(a.value - b.value)/this.normInt;
    }

    @Override
    public double evaluate_withSpeed(Spot a,
                           Spot b,
                           PartitionedGraph frames,
                           int dimension)
    {
        // Distance
        double maxAllowedDistance = 10;             // px
        double dt                 = 1.0;            // frame interval
        // TODO find frame interval !!!
        SimpleDistanceCost dist = new SimpleDistanceCost(costMax);

        double distance = dist.evaluate(a, b);
        if (distance > maxAllowedDistance)
            return Double.POSITIVE_INFINITY;

        // intensity
        double intensityDiff = Math.abs(a.value - b.value);

        // direction
        DirectionVector dirA = TrackingFunctions.findDirection(a, frames, dimension);
        DirectionVector dirB = TrackingFunctions.findDirection(b, frames, dimension);
        double angularPenalty = 1.0 - Math.max(0.0, dirA.cosineSimilarity(dirB)); // 0 (parallel) … 2 (opposite)

        // speed
        double speedA = dirA.norm() / dt; // Speed at frame t  (|delta r| / delta t)
        double dx     = b.x - a.x;
        double dy     = b.y - a.y;
        double speedB = Math.sqrt(dx*dx + dy*dy ) / dt; // speed from a to b

        // normalise speed change (0 = perfect, 1 = +100 %, 2 = −100 %, …)
        double speedPenalty = Math.abs(speedB - speedA) / (speedA + 1e-6); // avoid /0

        // FINAL COST
        double cost = lambda *  distance           / normDist  +          // distance
                      gamma  *  angularPenalty                   +          // angle
                      kappa  *  speedPenalty                     +          // speed
                      (1.0 - lambda - gamma - kappa) * intensityDiff / normInt; // intensity

        return cost;
    }


    @Override
    public boolean validate(Spot a, Spot b, PartitionedGraph frames, int dimension) {
        if (a == null) return false;
        if (b == null) return false;
        return evaluate(a, b, frames, dimension) < costMax;
    }
}



