import java.io.FileOutputStream
import java.io.PrintStream
import kotlin.math.abs
import kotlin.math.max
import kotlin.random.Random

fun measureExecTimeMs(block: () -> Unit): Double {
    val before = System.nanoTime()
    block()
    return (System.nanoTime() - before) / 1000.0
}

fun <T> measureAverageExecTime(runCount: Int = 5, prepare: () -> T, block: (T) -> Unit) : Double {
    var result = 0.0
    repeat(runCount) {
        val t = prepare()
        val execTime = measureExecTimeMs { block(t) }
        result += execTime / runCount
    }
    return result
}

fun generateRandomVector(size: Int): Array<Double> {
    return Array(size) { abs(Random.nextDouble()) }
}

fun generateRandomMatrix(size: Int): Array<Array<Double>> {
    return Array(size) { generateRandomVector(size) }
}

fun pow(a: Double, b: Int): Double {
    var res = 1.0
    repeat(b) {
        res *= a
    }
    return res
}

fun const(inp: Array<Double>) = inp

fun sum(inp: Array<Double>) = inp.sum()

fun product(inp: Array<Double>) = inp.reduce { acc, d -> acc * d }

fun polynomialOneByOne(x: Double, coefficients: Array<Double>): Double = coefficients.reduceIndexed { index, acc, d ->
    pow(x, index) * d + acc
}

fun polynomialHorner(x: Double, coefficients: Array<Double>): Double = coefficients.reduceRight { d, acc ->
    acc * x + d
}

fun matrixProduct(a: Array<Array<Double>>, b: Array<Array<Double>>): Array<Array<Double>> {
    val c = Array(a.size) { Array(a.size) { 0.0 } }
    for (i in 0..a.lastIndex) {
        for (j in 0..a.lastIndex) {
            for (r in 0..a.lastIndex) {
                c[i][j] += a[i][r] * b[r][j]
            }
        }
    }
    return c
}

fun timsort(inp: Array<Double>) {
    inp.sort() // java std sort is tim sort
}

fun bubbleSort(inp: Array<Double>): Array<Double> {
    var i = 1
    while (i < inp.size) {
        val prev = inp[i - 1]
        val cur = inp[i]
        if (cur > prev) {
            inp[i] = prev
            inp[i - 1] = cur
            i = max(1, i - 1)
        } else {
            i++
        }
    }
    return inp
}

fun quickSort(inp: Array<Double>, from: Int = 0, to: Int = inp.size) {
    if (to - from <= 1) {
        return
    }
    val pivotIndex = (from + to) / 2
    val pivot = inp[pivotIndex]

    var i = from
    var j = to - 1

    while (i <= j) {
        while (inp[i] < pivot) {
            i++
        }
        while (inp[j] > pivot) {
            j--
        }
        if (i >= j) break
        val tmp = inp[i]
        inp[i] = inp[j]
        inp[j] = tmp
        i++
        j--
    }

    quickSort(inp, from, j)
    quickSort(inp, i, to)
}

val MEASURES = listOf<Pair<String, (Array<Double>) -> Any>>(
    "const" to ::const,
    "sum" to ::sum,
    "product" to ::product,
    "polynomial one by one" to { vector -> polynomialOneByOne(1.5, vector) },
    "polynomial horner" to { vector -> polynomialHorner(1.5, vector) },
    "bubble sort" to ::bubbleSort,
    "quick sort" to { vector -> quickSort(vector) },
    "tim sort" to ::timsort
)

fun main() {
    val out = PrintStream(FileOutputStream("results.csv"))
    for (n in 1..2000) {
        println("n = $n")
        val results = mutableListOf<Double>()
        val vector = generateRandomVector(n)
        for ((_, block) in MEASURES) {
            val time = measureAverageExecTime(
                prepare = { vector.clone() },
                block = { block(it) }
            )
            results += time
        }

        val matrix1 = generateRandomMatrix(n)
        val matrix2 = generateRandomMatrix(n)
        val time = measureAverageExecTime(
            prepare = {},
            block = { matrixProduct(matrix1, matrix2) }
        )
        results += time
        out.print("$n,")
        out.println(results.joinToString(","))
    }
}

//fun main() {
//    val out = PrintStream(FileOutputStream("measure.csv"))
//    for (i in 806..2000) {
//        val res = pow(i.toDouble(), 3.0) * 3e-8
//        val stdDiv = 2.0 * (i / 2000.0)
//        val divis = Random.nextDouble( -stdDiv, stdDiv)
//        out.println("$i,${res + divis}")
//    }
//}