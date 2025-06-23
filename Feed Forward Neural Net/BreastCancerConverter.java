import java.util.*;

public class BreastCancerConverter {

    // 1) class (2 values)
    private static final String[] CLASS_VALUES = {
            "no-recurrence-events",
            "recurrence-events"
    };

    // 2) age (9 values)
    private static final String[] AGE_VALUES = {
            "10-19","20-29","30-39","40-49","50-59",
            "60-69","70-79","80-89","90-99"
    };

    // 3) menopause (3 values)
    private static final String[] MENOPAUSE_VALUES = {
            "lt40","ge40","premeno"
    };

    // 4) tumor-size (12 values)
    private static final String[] TUMOR_SIZE_VALUES = {
            "0-4","5-9","10-14","15-19","20-24","25-29",
            "30-34","35-39","40-44","45-49","50-54","55-59"
    };

    // 5) inv-nodes (13 values)
    private static final String[] INV_NODES_VALUES = {
            "0-2","3-5","6-8","9-11","12-14","15-17",
            "18-20","21-23","24-26","27-29","30-32","33-35","36-39"
    };

    // 6) node-caps (2 values)
    private static final String[] NODE_CAPS_VALUES = {
            "yes","no"
    };

    // 7) deg-malig (3 values)
    private static final String[] DEG_MALIG_VALUES = {
            "1","2","3"
    };

    // 8) breast (2 values)
    private static final String[] BREAST_VALUES = {
            "left","right"
    };

    // 9) breast-quad (5 values)
    private static final String[] BREAST_QUAD_VALUES = {
            "left-up","left-low","right-up","right-low","central"
    };

    // 10) irradiat (2 values)
    private static final String[] IRRADIAT_VALUES = {
            "yes","no"
    };

    // Offsets in the final 53-dim one-hot vector:
    //   class        -> 0..1    (2)
    //   age          -> 2..10   (9)
    //   menopause    -> 11..13  (3)
    //   tumor-size   -> 14..25  (12)
    //   inv-nodes    -> 26..38  (13)
    //   node-caps    -> 39..40  (2)
    //   deg-malig    -> 41..43  (3)
    //   breast       -> 44..45  (2)
    //   breast-quad  -> 46..50  (5)
    //   irradiat     -> 51..52  (2)
    //
    // Total = 53 bits (index range 0..52)

    // Encode a single line (10 space-separated fields) into a 53-length int[].
    public static int[] encodeLine(String line) {
        // Split line into exactly 10 tokens:
        String[] tokens = line.trim().split("\\s+");
        if (tokens.length != 10) {
            throw new IllegalArgumentException(
                    "Expected 10 fields, got: " + Arrays.toString(tokens));
        }
        int[] oneHot = new int[53];  // all zeros by default

        int offset = 0;

        // 1) class
        offset = setOneHot(oneHot, offset, CLASS_VALUES, tokens[0]);

        // 2) age
        offset = setOneHot(oneHot, offset, AGE_VALUES, tokens[1]);

        // 3) menopause
        offset = setOneHot(oneHot, offset, MENOPAUSE_VALUES, tokens[2]);

        // 4) tumor-size
        offset = setOneHot(oneHot, offset, TUMOR_SIZE_VALUES, tokens[3]);

        // 5) inv-nodes
        offset = setOneHot(oneHot, offset, INV_NODES_VALUES, tokens[4]);

        // 6) node-caps
        offset = setOneHot(oneHot, offset, NODE_CAPS_VALUES, tokens[5]);

        // 7) deg-malig
        offset = setOneHot(oneHot, offset, DEG_MALIG_VALUES, tokens[6]);

        // 8) breast
        offset = setOneHot(oneHot, offset, BREAST_VALUES, tokens[7]);

        // 9) breast-quad
        offset = setOneHot(oneHot, offset, BREAST_QUAD_VALUES, tokens[8]);

        // 10) irradiat
        offset = setOneHot(oneHot, offset, IRRADIAT_VALUES, tokens[9]);

        // offset should end up at 53
        return oneHot;
    }

    // Helper: sets the matching index to 1 for the current categorical field,
    // then returns the new offset after we consume that many columns.
    private static int setOneHot(int[] array, int start, String[] possibleValues, String token) {
        int index = indexOf(possibleValues, token);
        if (index < 0) {
            throw new IllegalArgumentException("Unrecognized token \"" + token
                    + "\" in " + Arrays.toString(possibleValues));
        }
        array[start + index] = 1;
        return start + possibleValues.length; // move offset
    }

    // Returns the position of 'token' in 'arr', or -1 if not found
    private static int indexOf(String[] arr, String token) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(token)) {
                return i;
            }
        }
        return -1;
    }

    public static void main(String[] args) {
        // Example usage with 4 lines, or read from System.in
        // For brevity, we'll just hardcode the 4 examples from the problem:
        List<String> lines = Arrays.asList(
                "no-recurrence-events 20-29 ge40 15-19 12-14 yes 3 right left-low no",
                "recurrence-events 60-69 premeno 20-24 9-11 yes 3 right left-up yes",
                "recurrence-events 10-19 lt40 0-4 21-23 no 3 left central yes",
                "recurrence-events 40-49 premeno 30-34 36-39 yes 3 left right-low no"
        );

        // Encode each line:
        for (String line : lines) {
            int[] encoded = encodeLine(line);
            // Print them out (comma-separated or space-separated):
            System.out.println(Arrays.toString(encoded));
        }
    }
}
