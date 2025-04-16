// GeneticAlgorithm.cs (Refactored with shared crossover for torque & steering)
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.Collections;
using System.IO;

public class GeneticAlgorithm : MonoBehaviour
{
    public int populationSize = 50;
    public int initialGeneLength = 400;
    public float mutationRate = 0.01f;
    public float crossoverRate = 0.7f;
    public int generations = 10000;
    public bool dynamicGeneLength = true;
    public bool useSegmentCrossover = true;
    [SerializeField] private GameObject robotPrefab;

    private List<List<float>> torquePopulation;
    private List<List<float>> steeringPopulation;
    private List<float> torqueFitnessScores;
    private List<float> steeringFitnessScores;
    private List<RobotController> robotInstances;
    private List<bool> activeIndividuals;
    private int currentStep = 0;
    private int currentGeneration = 0;
    private int currentGeneLength;
    private int freezeIndexTorque = 0;
    private int freezeIndexSteering = 0;
    private bool isCoolDown = false;
    private int maxCoolDownSteps = 500;
    private int coolDownStep = 0;
    private List<float> possibleValues = new List<float>();
    private int steadyGenerations = 0;
    private const int steadyThreshold = 3;
    private const float trimPercent = 0.1f;
    private int previousGeneLength;
    private int[] lastUsedGeneIndex;
    private Queue<(List<float> torque, List<float> steer)> bestHistory = new();
    private int bestHistoryLimit = 5;
    // New field for CSV writing
    private StreamWriter csvWriter;



    private void Start()
    {
        currentGeneLength = initialGeneLength;
        previousGeneLength = currentGeneLength;
        lastUsedGeneIndex = new int[populationSize];

        foreach (var listener in FindObjectsOfType<AudioListener>())
            if (listener != GetComponent<AudioListener>()) listener.enabled = false;

        InitializePossibleValues();
        InitializePopulation();
        ResetGeneration();
    }

    void InitializePossibleValues()
    {
        for (float v = -1f; v <= 1f; v += 0.04f)
            possibleValues.Add((float)System.Math.Round(v, 2));
    }

    void InitializePopulation()
    {
        torquePopulation = new();
        steeringPopulation = new();
        torqueFitnessScores = new(new float[populationSize]);
        steeringFitnessScores = new(new float[populationSize]);
        activeIndividuals = new(new bool[populationSize]);

        for (int i = 0; i < populationSize; i++)
        {
            torquePopulation.Add(CreateGeneSequence(currentGeneLength, (possibleValues.Count / 2) - 1, (int)(possibleValues.Count * 0.65)));
            steeringPopulation.Add(CreateGeneSequence(currentGeneLength));
            activeIndividuals[i] = true;
        }
    }

    List<float> CreateGeneSequence(int len, int start = 0, int end = -1)
    {
        List<float> seq = new();
        end = end < 0 ? possibleValues.Count : end;
        for (int i = 0; i < len; i++)
            seq.Add(possibleValues[Random.Range(start, end)]);
        return seq;
    }

    void InitializeRobots()
    {
        if (robotInstances != null)
            foreach (var r in robotInstances)
                if (r != null) Destroy(r.gameObject);

        robotInstances = new List<RobotController>();
        GameObject environment = GameObject.Find("Environment");

        for (int i = 0; i < populationSize; i++)
        {
            GameObject obj = Instantiate(robotPrefab, new Vector3(195.6539f, 0.6679955f, -105f), Quaternion.Euler(0f, 180f, 0f));
            obj.transform.SetParent(environment.transform);
            obj.layer = LayerMask.NameToLayer("Robot");
            RobotController rc = obj.GetComponent<RobotController>();
            rc.InitializeForGA(this, i);

            robotInstances.Add(rc);
            StartCoroutine(DelayedSetIndividual(rc, i));
            
        }
    }

    private IEnumerator DelayedSetIndividual(RobotController rc, int i)
    {
        yield return new WaitForFixedUpdate(); // Wait for 1 physics frame

        List<Vector2> combined = new List<Vector2>();
        for (int j = 0; j < torquePopulation[i].Count; j++)
            combined.Add(new Vector2(torquePopulation[i][j], steeringPopulation[i][j]));

        rc.SetIndividual(combined);

        // Optional: apply neutral control once before stepping
        rc.ManualApplyControl(0f, 0f);
    }

    void FixedUpdate()
    {
        if (isCoolDown)
        {
            if (++coolDownStep > maxCoolDownSteps)
            {
                isCoolDown = false; coolDownStep = 0;
            }
            for (int i = 0; i < populationSize; i++)
            {
                robotInstances[i].ManualApplyControl(0f, 0f);
            }

            return;
        }
        if (currentGeneration >= generations) return;

        if (currentStep < currentGeneLength || !AllIndividualsDone())
        {
            for (int i = 0; i < populationSize; i++)
            {
                if (activeIndividuals[i] && currentStep < torquePopulation[i].Count)
                {
                    float torque = torquePopulation[i][currentStep] * 500f;
                    float steer = steeringPopulation[i][currentStep] * 40f;
                    robotInstances[i].ManualApplyControl(torque, steer);

                    float appliedTorque = robotInstances[i].GetCurrentMotorTorque();
                    float appliedSteering = robotInstances[i].GetCurrentSteerAngle();
                    float reward = robotInstances[i].GetSteeringReward();

                    // Log data to CSV immediately
                    csvWriter.WriteLine($"{currentGeneration},{i},{currentStep},{torquePopulation[i][currentStep]},{steeringPopulation[i][currentStep]},{appliedTorque},{appliedSteering},{reward}");

                    lastUsedGeneIndex[i] = currentStep;

                }
                else if (activeIndividuals[i] && dynamicGeneLength)
                {
                    (float, string) ors = robotInstances[i].CheckOrientationSensor();
                    ExtendIndividual(i, ors.Item1, robotInstances[i].GetRoad());
                    float torque = torquePopulation[i][currentStep] * 500f;
                    float steer = steeringPopulation[i][currentStep] * 40f;
                    robotInstances[i].ManualApplyControl(torque, steer);

                    float appliedTorque = robotInstances[i].GetCurrentMotorTorque();
                    float appliedSteering = robotInstances[i].GetCurrentSteerAngle();
                    float fitness = steeringFitnessScores[i]; // Current fitness (may be 0 until finalized)

                    // Log data to CSV immediately
                    csvWriter.WriteLine($"{currentGeneration},{i},{currentStep},{torquePopulation[i][currentStep]},{steeringPopulation[i][currentStep]},{appliedTorque},{appliedSteering},{fitness}");

                    lastUsedGeneIndex[i] = currentStep;

                }
            }
            currentStep++;

            for (int i = 0; i < populationSize; i++)
                if (activeIndividuals[i]) robotInstances[i].UpdateFitness(currentStep > 1000);
        }
        else if (!activeIndividuals.Contains(true))
        {
            csvWriter.Close();
            Debug.Log($"Generation {currentGeneration} data logged to {Path.Combine(Application.persistentDataPath, $"generation_{currentGeneration}.csv")}");
            EvolveBothPopulations();
            isCoolDown = true;
            currentGeneration++;
            currentStep = 0;
            ResetGeneration();
        }
    }

    private bool AllIndividualsDone() => !activeIndividuals.Contains(true);

    void ExtendIndividual(int index, float ors, int turn = 0)
    {
        int end = (int)(possibleValues.Count * 0.60f);
        int start = (int)(possibleValues.Count / 2);
        if (ors <= -2f)
        {
            end = (int)possibleValues.Count;
            start = (int)(possibleValues.Count * 0.9f);
        }
        float t = possibleValues[Random.Range(start, end)];

        // Debug.Log($"start: {start}, end: {end}, ors: {ors}, torque: {t}");

        // int val1 = turn > 0 ? (int)(possibleValues.Count * 0.25f) : 0;
        // int val2 = turn < 0 ? (int)(possibleValues.Count * 0.75f) : possibleValues.Count;

        int val1 = 0;
        int val2 = possibleValues.Count;

        if (turn == 0)
        {
            if (ors <= -2f)
            {
                val1 = (int)(possibleValues.Count * 0.35f);
                val2 = (int)(possibleValues.Count * 0.65f);
            }
            else if (ors >= 2f)
            {

                val1 = (int)(possibleValues.Count * 0.35f);
                val2 = (int)(possibleValues.Count * 0.65f);
            }
        }
        else if (turn == 1)
        {
            val1 = (int)(possibleValues.Count * 0.25f);
            val2 = possibleValues.Count;
        }
        else if (turn == 2)
        {
            val1 = 0;
            val2 = (int)(possibleValues.Count * 0.75f);
        }
        else if (turn == 3)
        {
            val1 = (int)(possibleValues.Count * 0.25f);
            val2 = possibleValues.Count;
        }


        float s = possibleValues[Random.Range(val1, val2)];

        // Debug.Log($"ExtendIndividual steer: {s}, val1: {val1}, val2: {val2} torque: {t} turn: {turn}, end: {end}, total: {possibleValues.Count}, ors: {ors}");
        torquePopulation[index].Add(t);
        steeringPopulation[index].Add(s);
        currentGeneLength = Mathf.Max(currentGeneLength, torquePopulation[index].Count);
    }

    public void UpdateFitness(int index, float torqueFit, float steerFit, bool done)
    {
        if (done)
        {
            torqueFitnessScores[index] = torqueFit;
            steeringFitnessScores[index] = steerFit;
            activeIndividuals[index] = false;
            Debug.Log($"Finished car: {index}, fitness: {steerFit}, geneLength: {torquePopulation[index].Count}, currentStep: {currentStep}");
        }
    }

    void ResetGeneration()
    {
        activeIndividuals = new List<bool>(new bool[populationSize]);
        for (int i = 0; i < populationSize; i++) activeIndividuals[i] = true;

        // Open a new CSV file for this generation
        string filePath = Path.Combine(Application.persistentDataPath, $"generation_{currentGeneration}.csv");
        csvWriter = new StreamWriter(filePath);
        csvWriter.WriteLine("Generation,IndividualIndex,Step,GeneTorque,GeneSteering,AppliedTorque,AppliedSteering,Fitness");

        InitializeRobots();
    }

    void EvolveBothPopulations()
    {
        List<int> sorted = GetSortedIndices(steeringFitnessScores);
        var bestTorque = new List<float>(torquePopulation[sorted[0]]);
        var bestSteer = new List<float>(steeringPopulation[sorted[0]]);
        bestHistory.Enqueue((bestTorque, bestSteer));
        if (bestHistory.Count > bestHistoryLimit)
            bestHistory.Dequeue();
        float avg = Average(steeringFitnessScores);
        float best = Max(steeringFitnessScores);
        float diff = best - avg;
        bool trimming = false;

        // 🧠 Check if gene length was increased in this generation
        if (currentGeneLength > previousGeneLength)
        {
            Debug.Log("Gene length increased, skipping trim and resetting steadyGenerations.");
            steadyGenerations = 0;
            if (currentGeneLength >= 1500)
            {
                freezeIndexSteering = Mathf.Min(freezeIndexSteering + currentGeneLength / 10, currentGeneLength - 800);
            }
            else
            {
                freezeIndexSteering = Mathf.Min(freezeIndexSteering + currentGeneLength / 10, (int)(currentGeneLength / 3));
            }

            // freezeIndexSteering = 0;
            freezeIndexTorque = freezeIndexSteering;
            previousGeneLength = currentGeneLength; // Update to current
        }
        else if (avg >= 0.8f * best || (currentGeneLength >= 2200 && avg >= 0.7f * best))
        {
            steadyGenerations++;
            if (currentGeneLength >= 1500)
            {
               freezeIndexSteering = Mathf.Min(freezeIndexSteering + currentGeneLength / 10, currentGeneLength - 800);
            }
            else
            {
                freezeIndexSteering = Mathf.Min(freezeIndexSteering + currentGeneLength / 10, (int)(currentGeneLength / 3));
            }

            // freezeIndexSteering = 0;
            freezeIndexTorque = freezeIndexSteering;

            if ((steadyGenerations >= steadyThreshold && diff <= 2500f) || steadyGenerations >= 4)
            {
                // int trimMax = currentGeneLength >= 2500 ? 150 : 250;
                // int trimAmount = Mathf.Min(trimMax, Mathf.CeilToInt(currentGeneLength * trimPercent));
                int trimAmount = 200;
                trimming = true;


                Debug.Log($"🔥 Trimming last {trimAmount} genes from each individual (current geneLength: {currentGeneLength})");

                int bestCarIndex = sorted[0];

                for (int i = 0; i < populationSize; i++)
                {
                    if (i == bestCarIndex)
                    {
                        Debug.Log($"Best car: Not trimming: length: {torquePopulation[i].Count}: carIndex: {i}, bestcarIndex: {bestCarIndex}");
                        currentGeneLength = Mathf.Max(currentGeneLength, torquePopulation[i].Count);
                        continue;
                    }

                    int lastUsed = lastUsedGeneIndex[i];
                    int trimStart = Mathf.Max(0, lastUsed - trimAmount + 1); // e.g., trim 100 genes before last used
                    int trimCount = torquePopulation[i].Count - trimStart;

                    if (trimCount > 0 && trimStart < torquePopulation[i].Count)
                    {
                        torquePopulation[i].RemoveRange(trimStart, trimCount);
                        steeringPopulation[i].RemoveRange(trimStart, trimCount);
                        Debug.Log($"Lastused: {lastUsed}, trimStart: {trimStart}, trimCount:{trimCount}, lenghtaftertrim: {torquePopulation[i].Count}: carIndex: {i}, bestcarIndex: {bestCarIndex}");
                        currentGeneLength = Mathf.Max(currentGeneLength, torquePopulation[i].Count); // Don't shrink below initial
                    }
                }

                steadyGenerations = 0;
            }
        }
        else
        {
            steadyGenerations = 0;
        }


        Debug.Log($"Torque best: {Max(torqueFitnessScores)}, avg: {Average(torqueFitnessScores)}, generation: {currentGeneration}, geneLength: {currentGeneLength}");
        Debug.Log($"Steering best: {best}, avg: {avg}, generation: {currentGeneration}, geneLength: {currentGeneLength}, freezeIndex: {freezeIndexSteering}, steadyGenerations: {steadyGenerations}");

        CreateNewPopulationPair(ref torquePopulation, torqueFitnessScores, freezeIndexTorque,
                                ref steeringPopulation, steeringFitnessScores, freezeIndexSteering, sorted, trimming);
        previousGeneLength = currentGeneLength; // ✅ Always update this at the end
    }

    void CreateNewPopulationPair(ref List<List<float>> torquePop, List<float> torqueScores, int freezeTorque,
                                 ref List<List<float>> steerPop, List<float> steerScores, int freezeSteer, List<int> sorted, bool trimming)
    {
        int eliteCount = Mathf.Max(1, (int)(populationSize / 10));
        int poolSize = populationSize / 2;
        List<List<float>> newTorque = new();
        List<List<float>> newSteer = new();

        if (dynamicGeneLength && !trimming)
        {
            for (int i = 0; i < torquePop.Count; i++)
            {
                ExtendToLength(torquePop[i], currentGeneLength, torquePop[sorted[0]]);
                ExtendToLength(steerPop[i], currentGeneLength, steerPop[sorted[0]]);
            }
        }

        for (int i = 0; i < eliteCount; i++)
        {
            newTorque.Add(new List<float>(torquePop[sorted[i]]));
            newSteer.Add(new List<float>(steerPop[sorted[i]]));
        }

        foreach (var (torque, steer) in bestHistory)
        {
            newTorque.Add(new List<float>(torque));
            newSteer.Add(new List<float>(steer));
            if (newTorque.Count >= populationSize) break;
        }

        // 🔥 Generate at least 5 children from BEST + random in pool using segment crossover
        int childrenFromBest = 3;
        int bestIdx = sorted[0];
        newTorque.Add(new List<float>(torquePop[bestIdx]));
        newSteer.Add(new List<float>(steerPop[bestIdx]));
        newTorque.Add(new List<float>(torquePop[bestIdx]));
        newSteer.Add(new List<float>(steerPop[bestIdx]));
        newTorque.Add(new List<float>(torquePop[bestIdx]));
        newSteer.Add(new List<float>(steerPop[bestIdx]));

        for (int i = 0; i < childrenFromBest && newTorque.Count < populationSize; i++)
        {
            int mateIdx = sorted[Random.Range(1, eliteCount)]; // avoid best mating with itself

            int? pt1 = null, pt2 = null;
            if (useSegmentCrossover && torquePop[bestIdx].Count > freezeTorque + 2)
            {
                int len1 = torquePop[bestIdx].Count;
                int len2 = torquePop[mateIdx].Count;
                int minLen = Mathf.Min(len1, len2);

                if (useSegmentCrossover && minLen > freezeTorque + 2)
                {
                    int maxPt1 = minLen - 2;
                    pt1 = Random.Range(freezeTorque + 1, maxPt1 + 1);
                    pt2 = Random.Range(pt1.Value, minLen);
                }

            }

            SharedCrossover(torquePop[bestIdx], torquePop[mateIdx], out var c1T, out _, freezeTorque, pt1, pt2);
            SharedCrossover(steerPop[bestIdx], steerPop[mateIdx], out var c1S, out _, freezeSteer, pt1, pt2);

            Mutate(c1T, freezeTorque);
            Mutate(c1S, freezeSteer);

            // ExtendToLength(c1T, currentGeneLength, torquePop[bestIdx]);
            // ExtendToLength(c1S, currentGeneLength, steerPop[bestIdx]);

            newTorque.Add(c1T);
            newSteer.Add(c1S);
        }

        while (newTorque.Count < populationSize)
        {
            int idx1 = sorted[Random.Range(0, poolSize)];
            int idx2 = sorted[Random.Range(0, poolSize)];

            int? pt1 = null, pt2 = null;
            if (useSegmentCrossover && torquePop[idx1].Count > freezeTorque + 2)
            {
                int len1 = torquePop[idx1].Count;
                int len2 = torquePop[idx2].Count;
                int minLen = Mathf.Min(len1, len2);

                if (useSegmentCrossover && minLen > freezeTorque + 2)
                {
                    int maxPt1 = minLen - 2;
                    pt1 = Random.Range(freezeTorque + 1, maxPt1 + 1);
                    pt2 = Random.Range(pt1.Value, minLen);
                }

            }

            SharedCrossover(torquePop[idx1], torquePop[idx2], out var c1T, out var c2T, freezeTorque, pt1, pt2);
            SharedCrossover(steerPop[idx1], steerPop[idx2], out var c1S, out var c2S, freezeSteer, pt1, pt2);

            Mutate(c1T, freezeTorque);
            Mutate(c2T, freezeTorque);
            Mutate(c1S, freezeSteer);
            Mutate(c2S, freezeSteer);

            // ExtendToLength(c1T, currentGeneLength, torquePop[sorted[0]]);
            // ExtendToLength(c2T, currentGeneLength, torquePop[sorted[0]]);
            // ExtendToLength(c1S, currentGeneLength, steerPop[sorted[0]]);
            // ExtendToLength(c2S, currentGeneLength, steerPop[sorted[0]]);

            newTorque.Add(c1T);
            newSteer.Add(c1S);
            if (newTorque.Count < populationSize)
            {
                newTorque.Add(c2T);
                newSteer.Add(c2S);
            }
        }

        torquePop = newTorque;
        steerPop = newSteer;
    }

    void SharedCrossover(List<float> p1, List<float> p2, out List<float> c1, out List<float> c2, int freezeIdx, int? fixedPt1 = null, int? fixedPt2 = null)
    {
        c1 = new List<float>(p1);
        c2 = new List<float>(p2);

        if (useSegmentCrossover && p1.Count > freezeIdx + 2 && Random.value < crossoverRate)
        {
            int pt1 = fixedPt1 ?? Random.Range(freezeIdx + 1, p1.Count - 2);
            int pt2 = fixedPt2 ?? Random.Range(pt1, p1.Count - 1);

            for (int i = pt1; i <= pt2; i++)
            {
                float tmp = c1[i];
                c1[i] = c2[i];
                c2[i] = tmp;
            }
        }
    }

    void Mutate(List<float> individual, int freezeIdx)
    {
        for (int i = freezeIdx; i < individual.Count; i++)
        {
            float dynamicRate = mutationRate;
            // if (i > (individual.Count * .9f)) dynamicRate *= 10f;
            if (Random.value < dynamicRate)
                individual[i] = Mathf.Clamp(individual[i] + Random.Range(-0.1f, 0.1f), -1f, 1f);
        }
    }

    void ExtendToLength(List<float> individual, int targetLength, List<float> fallbackSource)
    {
        int currentLength = individual.Count;
        for (int i = currentLength; i < targetLength; i++)
        {
            float gene = (i < fallbackSource.Count) ? fallbackSource[i] : possibleValues[Random.Range(0, possibleValues.Count)];
            individual.Add(gene);
        }
    }

    float Max(List<float> values) => Mathf.Max(values.ToArray());
    float Average(List<float> values) => values.Count == 0 ? 0f : values.Sum() / values.Count;
    List<int> GetSortedIndices(List<float> scores)
    {
        List<int> idx = new();
        for (int i = 0; i < scores.Count; i++) idx.Add(i);
        idx.Sort((a, b) => scores[b].CompareTo(scores[a]));
        return idx;
    }
}
