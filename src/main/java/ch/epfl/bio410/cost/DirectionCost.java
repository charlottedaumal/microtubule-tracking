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

    /** normalization distance */
    private double normDist = 1;

    /** normalization intensity */
    private double normInt = 1;


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
     * @param dimension The width and height in pixel of the image area considered.
     * @return A non-negative double representing the directional cost.
     */
    @Override
    public double evaluate(Spot a, Spot b, PartitionedGraph frames, int dimension) {
        double maxAllowedDistance = 10;

        double intensityDiff = Math.abs(a.value - b.value);
        double distance = a.distance(b);

        DirectionVector dira = TrackingFunctions.findDirection(a, frames, dimension);
        DirectionVector dirb = TrackingFunctions.findDirection(b, frames, dimension);
        double directionSimilarity = dira.cosineSimilarity(dirb); // value between -1 and 1

        if (distance > maxAllowedDistance) return Double.POSITIVE_INFINITY;

        // Combine into a cost (lower is better)
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
     * @param dimension The width and height in pixel of the image area considered.
     * @return A non-negative double representing the directional cost.
     */
    @Override
    public double evaluate_withSpeed(Spot a, Spot b, PartitionedGraph frames, int dimension) {
        // Distance
        double maxAllowedDistance = 10;             // px
        double dt                 = 1.0;            // frame interval
        // TODO find frame interval !!!

        double distance = a.distance(b);
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


    /**
     * This method validates whether the directional link between two spots is allowed when speed is not considered during
     * cost evaluation.
     *
     * @param a The source Spot.
     * @param b The target Spot.
     * @param frames The PartitionedGraph structure that contains the temporal relationships between frames.
     * @param dimension The number of spatial dimensions (2 or 3).
     * @return true if the link from a to b is valid; false otherwise.
     */
    @Override
    public boolean validate(Spot a, Spot b, PartitionedGraph frames, int dimension) {
        if (a == null) return false;
        if (b == null) return false;
        return evaluate(a, b, frames, dimension) < costMax;
    }


    /**
     * This method validates whether the directional link between two spots is allowed when speed is considered during
     * cost evaluation.
     *
     * @param a The source Spot.
     * @param b The target Spot.
     * @param frames The PartitionedGraph structure that contains the temporal relationships between frames.
     * @param dimension The number of spatial dimensions (2 or 3).
     * @return true if the link from a to b is valid; false otherwise.
     */
    @Override
    public boolean validate_withSpeed(Spot a, Spot b, PartitionedGraph frames, int dimension) {
        if (a == null) return false;
        if (b == null) return false;
        return evaluate_withSpeed(a, b, frames, dimension) < costMax;
    }
}



