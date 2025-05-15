package ch.epfl.bio410.cost;

import ch.epfl.bio410.graph.Spot;

/**
 * This class implements the "SimpleDistanceCost" algorithm for tracking particles.
 * It implements the "AbstractCost" interface to benefit from the generic methods "evaluate" and "validate"
 */
public class SimpleDistanceCost implements AbstractCost {

    private double distmax;
    public SimpleDistanceCost(double distmax) {
        this.distmax = distmax;
    }

    @Override
    public double evaluate(Spot a, Spot b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    @Override
    public boolean validate(Spot a, Spot b) {
        if (a == null) return false;
        if (b == null) return false;
        return evaluate(a, b) < distmax;
    }
}
