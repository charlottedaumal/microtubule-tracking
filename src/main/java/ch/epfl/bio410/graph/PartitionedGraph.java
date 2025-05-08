package ch.epfl.bio410.graph;

import ij.ImagePlus;
import ij.gui.Line;
import ij.gui.OvalRoi;
import ij.gui.Overlay;

import java.util.ArrayList;


/**
 * Class implementing a "PartitionedGraph" object. A "PartitionedGraph" object is a list of "Spots" objects,
 * with additional methods to draw the tracking graph of particles.
 */
public class PartitionedGraph extends ArrayList<Spots> {

    public Spots getPartitionOf(Spot spot) {
        for (Spots spots : this) {
            for (Spot test : spots) {
                if (spot.equals(test))
                    return spots;
            }
        }
        return null;
    }

    public Spots createPartition(Spot spot) {
        Spots spots = new Spots();
        spots.add(spot);
        add(spots);
        return spots;
    }

    public Overlay drawSpots(ImagePlus imp) {
        Overlay overlay = imp.getOverlay();
        if (overlay == null) overlay = new Overlay();
        int radius = 5;
        for(Spots spots : this) {
            for(Spot spot : spots) {
                double xp = spot.x + 0.5 - radius;
                double yp = spot.y + 0.5 - radius;
                OvalRoi roi = new OvalRoi(xp, yp, 2 * radius, 2 * radius);
                roi.setPosition(spot.t + 1); // display roi in one frqme
                roi.setStrokeColor(spots.color);
                roi.setStrokeWidth(1);
                overlay.add(roi);
            }
        }
        ImagePlus out = imp.duplicate();
        out.setTitle("Spots " + imp.getTitle() );
        out.show();
        out.getProcessor().resetMinAndMax();
        out.setOverlay(overlay);
        return overlay;
    }


    public Overlay drawLines(ImagePlus imp) {
        Overlay overlay = imp.getOverlay();
        if (overlay == null) overlay = new Overlay();
        int radius = 3;
        for (Spots spots : this) {
            if (spots.isEmpty()) break;

            for (int i = 1; i < spots.size(); i++) {
                Spot spot = spots.get(i);
                Spot prev = spots.get(i - 1);
                Line line = new Line(prev.x + 0.5, prev.y + 0.5, spot.x + 0.5, spot.y + 0.5);
                line.setStrokeColor(spots.color);
                line.setStrokeWidth(2);
                overlay.add(line);

                OvalRoi roi1 = new OvalRoi(spot.x + 0.5 - radius, spot.y + 0.5 - radius, 2 * radius, 2 * radius);
                roi1.setPosition(spot.t + 1); // display roi in one frame
                roi1.setFillColor(spots.color);
                roi1.setStrokeWidth(1);
                overlay.add(roi1);

                OvalRoi roi2 = new OvalRoi(prev.x + 0.5 - radius, prev.y + 0.5 - radius, 2 * radius, 2 * radius);
                roi2.setPosition(prev.t + 1); // display roi in one frame
                roi2.setFillColor(spots.color);
                roi2.setStrokeWidth(1);
                overlay.add(roi2);
            }
        }
        ImagePlus out = imp.duplicate();
        out.setTitle("Trajectories " + imp.getTitle() );
        out.show();
        out.getProcessor().resetMinAndMax();
        out.setOverlay(overlay);
        return overlay;
    }

}

