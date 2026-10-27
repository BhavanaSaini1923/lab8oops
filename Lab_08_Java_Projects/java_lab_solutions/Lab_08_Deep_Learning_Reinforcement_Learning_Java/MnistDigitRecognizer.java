import java.io.*;
import java.util.*;

/**
 * Lab 08, Project 1: a compact one-hidden-layer neural network for MNIST CSV.
 * CSV format: label,pixel1,...,pixel784. Pixel values are normalized to [0,1].
 * This is an educational implementation; use a subset first to check the pipeline.
 */
public class MnistDigitRecognizer {
    static final int INPUT = 784, HIDDEN = 64, OUTPUT = 10;
    static final double LEARNING_RATE = 0.08;
    static final int EPOCHS = 3;
    static final Random random = new Random(7);
    static double[][] w1 = new double[HIDDEN][INPUT], w2 = new double[OUTPUT][HIDDEN];
    static double[] b1 = new double[HIDDEN], b2 = new double[OUTPUT];

    static {
        for (int h=0; h<HIDDEN; h++) for (int i=0; i<INPUT; i++) w1[h][i] = random.nextGaussian()*0.04;
        for (int o=0; o<OUTPUT; o++) for (int h=0; h<HIDDEN; h++) w2[o][h] = random.nextGaussian()*0.04;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage: java MnistDigitRecognizer mnist_train.csv mnist_test.csv");
            return;
        }
        List<Example> train = readCsv(args[0]);
        List<Example> test = readCsv(args[1]);
        System.out.println("Training rows: " + train.size() + " | Test rows: " + test.size());
        for (int epoch=1; epoch<=EPOCHS; epoch++) {
            Collections.shuffle(train, random);
            double loss=0;
            for (Example e : train) loss += trainOne(e);
            System.out.printf("Epoch %d/%d | average cross-entropy loss: %.4f%n", epoch, EPOCHS, loss/train.size());
        }
        int correct=0;
        for (Example e : test) if (predict(e.x)==e.label) correct++;
        double accuracy = test.isEmpty()?0:100.0*correct/test.size();
        System.out.printf("Test accuracy: %.2f%% (%d/%d)%n", accuracy, correct, test.size());
        saveModel("digit_model.txt");
        System.out.println("Saved model parameters to digit_model.txt");
    }

    static class Example {
        int label; double[] x;
        Example(int label, double[] x) { this.label=label; this.x=x; }
    }

    static List<Example> readCsv(String path) throws IOException {
        List<Example> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line; int lineNo=0;
            while ((line=br.readLine())!=null) {
                lineNo++;
                String[] p=line.trim().split(",");
                if (p.length != INPUT+1) {
                    if (lineNo==1) continue; // skip optional header
                    throw new IOException("Expected 785 columns at line " + lineNo + ", found " + p.length);
                }
                try {
                    int label=Integer.parseInt(p[0].trim());
                    if (label<0 || label>9) throw new NumberFormatException();
                    double[] x=new double[INPUT];
                    for(int i=0;i<INPUT;i++) x[i]=Double.parseDouble(p[i+1].trim())/255.0;
                    rows.add(new Example(label,x));
                } catch(NumberFormatException ex) {
                    if (lineNo==1) continue;
                    throw new IOException("Invalid numeric value at line " + lineNo, ex);
                }
            }
        }
        if(rows.isEmpty()) throw new IOException("No usable rows found in " + path);
        return rows;
    }

    static double trainOne(Example e) {
        double[] z1=new double[HIDDEN], a1=new double[HIDDEN], z2=new double[OUTPUT], y= new double[OUTPUT];
        for(int h=0;h<HIDDEN;h++) {
            z1[h]=b1[h];
            for(int i=0;i<INPUT;i++) z1[h]+=w1[h][i]*e.x[i];
            a1[h]=Math.max(0,z1[h]);
        }
        for(int o=0;o<OUTPUT;o++) {
            z2[o]=b2[o];
            for(int h=0;h<HIDDEN;h++) z2[o]+=w2[o][h]*a1[h];
        }
        double max=Arrays.stream(z2).max().orElse(0), sum=0;
        for(int o=0;o<OUTPUT;o++) { y[o]=Math.exp(z2[o]-max); sum+=y[o]; }
        for(int o=0;o<OUTPUT;o++) y[o]/=sum;
        double loss=-Math.log(Math.max(y[e.label],1e-12));

        double[] dz2=y.clone(); dz2[e.label]-=1;
        double[] da1=new double[HIDDEN];
        for(int h=0;h<HIDDEN;h++) {
            for(int o=0;o<OUTPUT;o++) da1[h]+=w2[o][h]*dz2[o];
        }
        for(int o=0;o<OUTPUT;o++) {
            for(int h=0;h<HIDDEN;h++) w2[o][h]-=LEARNING_RATE*dz2[o]*a1[h];
            b2[o]-=LEARNING_RATE*dz2[o];
        }
        for(int h=0;h<HIDDEN;h++) {
            double dz1=z1[h]>0?da1[h]:0;
            for(int i=0;i<INPUT;i++) w1[h][i]-=LEARNING_RATE*dz1*e.x[i];
            b1[h]-=LEARNING_RATE*dz1;
        }
        return loss;
    }

    static int predict(double[] x) {
        double[] a1=new double[HIDDEN], z2=new double[OUTPUT];
        for(int h=0;h<HIDDEN;h++) {
            double z=b1[h]; for(int i=0;i<INPUT;i++) z+=w1[h][i]*x[i];
            a1[h]=Math.max(0,z);
        }
        for(int o=0;o<OUTPUT;o++) {
            z2[o]=b2[o]; for(int h=0;h<HIDDEN;h++) z2[o]+=w2[o][h]*a1[h];
        }
        int best=0; for(int o=1;o<OUTPUT;o++) if(z2[o]>z2[best]) best=o;
        return best;
    }

    static void saveModel(String path) throws IOException {
        try(PrintWriter out=new PrintWriter(new FileWriter(path))) {
            out.println("MNIST_DNN_V1");
            for(double[] row:w1) for(double v:row) out.println(v);
            for(double v:b1) out.println(v);
            for(double[] row:w2) for(double v:row) out.println(v);
            for(double v:b2) out.println(v);
        }
    }
}
