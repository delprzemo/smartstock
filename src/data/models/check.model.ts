export class CheckModel {
    constructor(
        public input: number[] = [],
        public output: number = null,
        public calculatedOutput: number  = null,
        public error: number  = null,
        public isCorrectTrend: boolean  = null,
        public prediction: number = null) {

    }
}