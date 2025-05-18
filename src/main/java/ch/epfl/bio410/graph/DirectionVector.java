package ch.epfl.bio410.graph;

import static inference.LogFuncs.pi;

public class DirectionVector {
    public final double dx;
    public final double dy;
    public final Spot spot;

    public DirectionVector(double dx, double dy, Spot spot) {
        this.dx = dx;
        this.dy = dy;
        this.spot = spot;
    }


    public double norm() {
        return Math.sqrt(dx * dx + dy * dy);
    }

    public double dot(DirectionVector other) {
        return dx * other.dx + dy * other.dy;
    }

    public double cosineSimilarity(DirectionVector other) {
        double mag1 = norm();
        double mag2 = other.norm();
        if (mag1 == 0 || mag2 == 0) return 0;
        return dot(other) / (mag1 * mag2);
    }
}

