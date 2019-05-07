import { expect } from 'chai';
import * as tf from '@tensorflow/tfjs-node'
import { Tensor } from '@tensorflow/tfjs-node';

describe('Intro -tensorflow -  fun', () => {
    before(function () {
        this.timeout(1000000) // 10 second timeout for setup
    })

    // it("Basic trial", () => {

    //     // Tensor manipulation
    //     const t1 = tf.tensor(([1,2,3,4,2,4,6,8]), [2,4]);
    //     const squared = t1.square();


    // })

    // it("Simple prediction Y = 2X - 1", async function (done) {
    //     this.timeout(500000); // This works

    //     // declare models
    //     const model = tf.sequential(); //sequential model is any model where the outputs of one layer are the inputs to the next layer
    //     model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    //     // Specify loss and optimizer for model
    //     model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    //     // Prepare training data
    //     const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4, 10], [7, 1]);
    //     const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7, 19], [7, 1]);


    //     // Train the model
    //     await model.fit(xs, ys, {epochs: 1000}).then(async () => {
    //         // Use model to predict values
    //         const result = await model.predict(tf.tensor2d([20], [1,1])).toString();
    //         done();
    //     });
    // })

    // it("Simple prediction Sin", async function (done) {
    //     this.timeout(500000); // This works

    //     // declare models
    //     const model = tf.sequential(); //sequential model is any model where the outputs of one layer are the inputs to the next layer
    //     model.add(tf.layers.lstm({units: 8, inputShape: [1]}));
    //     model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    //     // Specify loss and optimizer for model
    //     model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    //     // Prepare training data
    //     const [xs, ys] = generateSinTrainingData(100);

    //     // Train the model
    //     await model.fit(xs, ys, { epochs: 1500 }).then(async () => {
    //         // Use model to predict values
    //         const result = await model.predict(tf.tensor2d([-0.9], [1, 1])).toString();
    //         done();
    //     });
    // })

    // it("Simple prediction Sin with two hidden layers", async function (done) {
    //     this.timeout(500000); // This works

    //     // declare models
    //     const model = tf.sequential(); //sequential model is any model where the outputs of one layer are the inputs to the next layer


    //     // inputShape [1] - one number is in input
    //     model.add(tf.layers.dense({ units: 5, inputShape: [1] })); // hidden layer
    //     model.add(tf.layers.dense({ units: 5, inputShape: [1] })); // hidden layer
    //     model.add(tf.layers.dense({ units: 1})); //output layer

    //     // Specify loss and optimizer for model
    //     model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    //     // Prepare training data
    //     const [xs, ys] = generateSinTrainingData(100);

    //     // Train the model
    //     await model.fit(xs, ys, { epochs: 1000 }).then(async () => {
    //         // Use model to predict values
    //         console.log(Math.sin(-0.9));
    //         const result = await model.predict(tf.tensor2d([-0.9], [1, 1])).toString();
    //         done();
    //     });
    // })


    it("Simple prediction Sin with LSTM", async function (done) {
        this.timeout(500000); // This works

        const size = 100;
        const windowSize = 5;
        const epochs = 40;
        const learningRate = 0.001;
        const layers = 2;


        // Prepare training data
        let sinTimeSeries = generateSinTimeSeries(100);
        let [input, output] = generateTimeSeriesInputOutpu(sinTimeSeries, 5);
        const trainingResult = await trainModel(input, output, size, windowSize, epochs, 
            learningRate, layers, () => {});
        const result = await Predict(input, 50, trainingResult.model);

    })


    async function trainModel(inputs: any[], outputs: any[], size: number, window_size: number, n_epochs: number, 
        learning_rate: number, n_layers: number, callback: Function) {
        const input_layer_shape = window_size;
        const input_layer_neurons = 20;

        const rnn_input_layer_features = 10;
        const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;

        const rnn_input_shape = [rnn_input_layer_features, rnn_input_layer_timesteps];
        const rnn_output_neurons = 20;

        const rnn_batch_size = window_size;

        const output_layer_shape = rnn_output_neurons;
        const output_layer_neurons = 1;

        const model = tf.sequential();

        inputs = inputs.slice(0, Math.floor(size / 100 * inputs.length));
        outputs = outputs.slice(0, Math.floor(size / 100 * outputs.length));

        const xs = tf.tensor2d(inputs, [inputs.length, inputs[0].length]).div(tf.scalar(10));
        const ys = tf.tensor2d(outputs, [outputs.length, 1]).reshape([outputs.length, 1]).div(tf.scalar(10));

        model.add(tf.layers.dense({ units: input_layer_neurons, inputShape: [input_layer_shape] }));
        model.add(tf.layers.reshape({ targetShape: rnn_input_shape }));

        var lstm_cells = [];
        for (let index = 0; index < n_layers; index++) {
            lstm_cells.push(tf.layers.lstmCell({ units: rnn_output_neurons }));
        }

        model.add(tf.layers.rnn({
            cell: lstm_cells,
            inputShape: rnn_input_shape, returnSequences: false
        }));

        model.add(tf.layers.dense({ units: output_layer_neurons, inputShape: [output_layer_shape] }));

        const opt_adam = tf.train.adam(learning_rate);
        model.compile({ optimizer: opt_adam, loss: 'meanSquaredError' });

        const hist = await model.fit(xs, ys,
            {
                batchSize: rnn_batch_size, epochs: n_epochs, callbacks: {
                    onEpochEnd: async (epoch, log) => { callback(epoch, log); }
                }
            });

        return { model: model, stats: hist };
    }

    function Predict(inputs: any[], size: number, model: tf.Sequential) {
        var inps = inputs.slice(Math.floor(size / 100 * inputs.length), inputs.length);

        inps = [[
        0.683261715,
        0.9835877454343449,
        0.3796077390275217,
        -0.573381872,
        -0.999206834,
    ]]

        const outps = (model.predict(tf.tensor2d(inps, [inps.length,
        inps[0].length]).div(tf.scalar(10))) as Tensor).mul(10);

        console.log(outps.toString());

        return Array.from(outps.dataSync());
    }


    function generateSinTrainingData(arrLength: number) {
        let input = [];
        let output = [];
        for (let i = 0; i < arrLength; i++) {
            let rand = Math.random();
            input.push(rand);
            output.push(Math.sin(rand));
        }

        const xs = tf.tensor2d(input, [input.length, 1]);
        const ys = tf.tensor2d(output, [output.length, 1]);

        return [xs, ys];
    }

    function generateSinTimeSeries(arrLength: number) {
        let result = [];
        for (let i = 0; i < arrLength; i++) {
            result.push(Math.sin(i));
        }

        return result;
    }

    function generateTimeSeriesInputOutpu(array: number[], windowSize: number) {
        let input = [];
        let output = [];
        for (let i = windowSize; i < array.length - 1; i++) {
            input.push(array.slice(i - windowSize, i));
            output.push(array[i]);
        }

        return [input, output];
    }
});