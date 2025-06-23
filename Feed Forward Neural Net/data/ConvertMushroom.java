package data;
import java.io.*;

public class ConvertMushroom {
    // For each of the 23 columns (column 0 = class, columns 1..22 = attributes),
    // the following array lists the possible categorical values in the order
    // in which you want the one‑hot vector to be arranged.
    static String[][] categories = {
        // 0. Class (we won’t use one-hot encoding for this; we’ll output "0:" or "1:" instead)
        {"e", "p"},
        // 1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
        {"b", "c", "x", "f", "k", "s"},
        // 2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
        {"f", "g", "y", "s"},
        // 3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
        {"n", "b", "c", "g", "r", "p", "u", "e", "w", "y"},
        // 4. bruises?: bruises=t, no=f
        {"t", "f"},
        // 5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
        {"a", "l", "c", "y", "f", "m", "n", "p", "s"},
        // 6. gill-attachment: attached=a, descending=d, free=f, notched=n
        {"a", "d", "f", "n"},
        // 7. gill-spacing: close=c, crowded=w, distant=d
        {"c", "w", "d"},
        // 8. gill-size: broad=b, narrow=n
        {"b", "n"},
        // 9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r,
        //               orange=o, pink=p, purple=u, red=e, white=w, yellow=y
        {"k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"},
        // 10. stalk-shape: enlarging=e, tapering=t
        {"e", "t"},
        // 11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
        {"b", "c", "u", "e", "z", "r", "?"},
        // 12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s
        {"f", "y", "k", "s"},
        // 13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s
        {"f", "y", "k", "s"},
        // 14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o,
        //                              pink=p, red=e, white=w, yellow=y
        {"n", "b", "c", "g", "o", "p", "e", "w", "y"},
        // 15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o,
        //                              pink=p, red=e, white=w, yellow=y
        {"n", "b", "c", "g", "o", "p", "e", "w", "y"},
        // 16. veil-type: partial=p, universal=u
        {"p", "u"},
        // 17. veil-color: brown=n, orange=o, white=w, yellow=y
        {"n", "o", "w", "y"},
        // 18. ring-number: none=n, one=o, two=t
        {"n", "o", "t"},
        // 19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l,
        //              none=n, pendant=p, sheathing=s, zone=z
        {"c", "e", "f", "l", "n", "p", "s", "z"},
        // 20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r,
        //                       orange=o, purple=u, white=w, yellow=y
        {"k", "n", "b", "h", "r", "o", "u", "w", "y"},
        // 21. population: abundant=a, clustered=c, numerous=n,
        //                scattered=s, several=v, solitary=y
        {"a", "c", "n", "s", "v", "y"},
        // 22. habitat: grasses=g, leaves=l, meadows=m, paths=p,
        //             urban=u, waste=w, woods=d
        {"g", "l", "m", "p", "u", "w", "d"}
    };

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: java ConvertMushroom <inputFile> <outputFile>");
            System.exit(1);
        }
        String inputFile = args[0];
        String outputFile = args[1];

        try (BufferedReader br = new BufferedReader(new FileReader(inputFile));
             PrintWriter pw = new PrintWriter(new FileWriter(outputFile))) {

            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                // Split the line on commas
                String[] tokens = line.split(",");
                if (tokens.length != categories.length) {
                    System.err.println("Skipping line (expected " + categories.length +
                            " tokens, got " + tokens.length + "): " + line);
                    continue;
                }

                // Process the class attribute specially:
                String classToken = tokens[0].trim();
                String classLabel;
                if (classToken.equals("e")) {
                    // edible: output as 0:
                    classLabel = "0:";
                } else if (classToken.equals("p")) {
                    // poisonous: output as 1:
                    classLabel = "1:";
                } else {
                    System.err.println("Unknown class token \"" + classToken +
                            "\" in line: " + line);
                    classLabel = "0:"; // default
                }

                // Build one-hot encoding for attributes 1 to 22 (tokens 1 to end)
                StringBuilder sb = new StringBuilder();
                sb.append(classLabel);
                // Loop from col = 1 because column 0 is the class (handled above)
                for (int col = 1; col < tokens.length; col++) {
                    String token = tokens[col].trim();
                    int pos = indexOf(categories[col], token);
                    if (pos == -1) {
                        System.err.println("Unknown token \"" + token +
                                "\" in column " + col + " (line skipped): " + line);
                        pos = 0; // default
                    }
                    // Build the one-hot vector for this attribute:
                    for (int j = 0; j < categories[col].length; j++) {
                        sb.append(j == pos ? "1" : "0");
                        if (j < categories[col].length - 1) {
                            sb.append(",");
                        }
                    }
                    // Append a comma between attributes (except after the last attribute)
                    if (col < tokens.length - 1) {
                        sb.append(",");
                    }
                }
                pw.println(sb.toString());
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    // Helper: returns the index of token in arr, or -1 if not found.
    private static int indexOf(String[] arr, String token) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(token)) {
                return i;
            }
        }
        return -1;
    }
}
