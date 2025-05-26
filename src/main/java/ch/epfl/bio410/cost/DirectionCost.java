package ch.epfl.bio410.cost;


import ch.epfl.bio410.graph.DirectionVector;
import ch.epfl.bio410.graph.PartitionedGraph;
import ch.epfl.bio410.graph.Spot;
import ch.epfl.bio410.graph.Spots;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.ZProjector;
import ch.epfl.bio410.utils.TrackingFunctions;


/**
 * The DirectionCost class implements the AbstractDirCost interface and provides methods to validate and evaluate the
 * directional relationship between two Spot objects in a tracking context. Here, it is used in a motion tracking framework
 * where the direction of movement between detections
 * influences the cost of linking them over time.
 */
public class DirectionCost implements AbstractDirCost {
    private double lambda = 0; // parameter for the intensity cost
    private double gamma = 0; // parameter for the direction cost
    private double kappa = 0; // parameter for the speed cost
    private double costMax = 0;
    private double normDist = 1; // normalization distance
    private double normInt = 1; // normalization intensity


    /**
     * Constructor of the class.
     *
     * @param imp The ImagePlus object.
     * @param costMax The maximum allowed cost for a connection.
     * @param lambda Weight for the distance-based component of the cost function.
     * @param gamma Weight for the directionality component of the cost function.
     * @param kappa Weight for the speed-based component of the cost function.
     *
     */
    public DirectionCost(ImagePlus imp, double costMax, double lambda, double gamma, double kappa) {
        this.lambda = lambda;
        this.gamma = gamma;
        this.kappa = kappa;
        this.costMax = costMax;
        int height = imp.getHeight();
        int width = imp.getWidth();
        this.normDist = Math.sqrt(height * height + width * width);
        this.normInt = ZProjector.run(imp,"max").getStatistics().max - ZProjector.run(imp,"min").getStatistics().min;
    }


    /**
     * This method computes the directional cost between two spots without considering speed.
     *
     * @param a The source Spot.
     * @param b The target Spot.
     * @param frames The PartitionedGraph containing frames.
     * @param dimension The size (in pixels) of the square search window used to look for the closest spot
     * in the previous frame.
     * @return A non-negative double representing the directional cost.
     */
    @Override
    public double evaluate(Spot a, Spot b, PartitionedGraph frames, int dimension) {
        // distance
        double maxAllowedDistance = 10;
        double distance = b.distance(a);
        if (distance > maxAllowedDistance) return Double.POSITIVE_INFINITY;

        // intensity
        double intensityDiff = Math.abs(a.value - b.value);

        // direction
        DirectionVector dira = TrackingFunctions.findDirection(a, frames, dimension);
        DirectionVector dirb = TrackingFunctions.findDirection(b, frames, dimension);
        double directionSimilarity = dira.cosineSimilarity(dirb); // value between -1 and 1

        // final cost
        return this.lambda * distance / this.normDist +
                this.gamma * (1 - Math.max(0, directionSimilarity)) +
                (1 - this.lambda - this.gamma)*intensityDiff/this.normInt;
    }


    /**
     * This method computes the directional cost between two spots including speed considerations.
     *
     * @param a The source Spot.
     * @param b The target Spot.
     * @param frames The PartitionedGraph containing frames.
     * @param dimension The size (in pixels) of the square search window used to look for the closest spot
     * in the previous frame.
     * @return A non-negative double representing the directional cost.
     */
    @Override
    public double evaluate_withSpeed(Spot a, Spot b, PartitionedGraph frames, int dimension) {
        // distance
        double maxAllowedDistance = 10; // in pixels
        double dt = 1.0; // frame interval
        // TODO find frame interval !!!
        double distance = a.distance(b);
        if (distance > maxAllowedDistance)
            return Double.POSITIVE_INFINITY;

        // intensity
        double intensityDiff = Math.abs(a.value - b.value);

        // direction
        DirectionVector dirA = TrackingFunctions.findDirection(a, frames, dimension);
        DirectionVector dirB = TrackingFunctions.findDirection(b, frames, dimension);
        double angularPenalty = 1.0 - Math.max(0.0, dirA.cosineSimilarity(dirB)); // 0 (parallel) and 2 (opposite)

        // speed
        double speedA = dirA.norm() / dt; // speed at frame t
        double speedB = b.distance(a) / dt; // speed from a to b
        // a large difference between previous speed and current speed is penalized
        double speedPenalty = Math.abs(speedB - speedA) / (speedA + 1e-6); // avoid division by zero

        // final cost
        return lambda *  distance / normDist  +
                gamma  *  angularPenalty +
                kappa  *  speedPenalty +
                (1.0 - lambda - gamma - kappa) * intensityDiff / normInt;
    }


    /**
     * This method validates whether the directional link between two spots is allowed when speed is not considered during
     * cost evaluation.
     *
     * @param a The source Spot.
     * @param b The target Spot.
     * @param frames The PartitionedGraph structure that contains the temporal relationships between frames.
     * @param dimension The size (in pixels) of the square search window used to look for the closest spot
     * in the previous frame.
     * @return true if the link from a to b is valid; false otherwise.
     */
    @Override
    public boolean validate(Spot a, Spot b, PartitionedGraph frames, int dimension) {
        // reject the connection if either of the spots is null
        if (a == null) return false;
        if (b == null) return false;
        // evaluate the cost of linking a to b; accept only if below the defined threshold
        return evaluate(a, b, frames, dimension) < costMax;
    }


    /**
     * This method validates whether the directional link between two spots is allowed when speed is considered during
     * cost evaluation.
     *
     * @param a The source Spot.
     * @param b The target Spot.
     * @param frames The PartitionedGraph structure that contains the temporal relationships between frames.
     * @param dimension The size (in pixels) of the square search window used to look for the closest spot
     * in the previous frame.
     * @return true if the link from a to b is valid; false otherwise.
     */
    @Override
    public boolean validate_withSpeed(Spot a, Spot b, PartitionedGraph frames, int dimension) {
        // reject the connection if either of the spots is null
        if (a == null) return false;
        if (b == null) return false;
        // evaluate the cost of linking a to b; accept only if below the defined threshold
        return evaluate_withSpeed(a, b, frames, dimension) < costMax;
    }
}



