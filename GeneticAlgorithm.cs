// GeneticAlgorithm.cs (Refactored with shared crossover for torque & steering)
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

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

    private void Start()
    {
        currentGeneLength = initialGeneLength;
        foreach (var listener in FindObjectsOfType<AudioListener>())
            if (listener != GetComponent<AudioListener>()) listener.enabled = false;

        InitializePossibleValues();
        InitializePopulation();
        InitializeRobots();
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

        for (int i = 0; i < populationSize; i++)
        {
            GameObject obj = Instantiate(robotPrefab, new Vector3(195.6539f, 0.6679955f, -105f), Quaternion.Euler(0f, 180f, 0f));
            obj.layer = LayerMask.NameToLayer("Robot");
            RobotController rc = obj.GetComponent<RobotController>();
            rc.InitializeForGA(this, i);

            List<Vector2> combined = new List<Vector2>();
            for (int j = 0; j < currentGeneLength; j++)
                combined.Add(new Vector2(torquePopulation[i][j], steeringPopulation[i][j]));

            rc.SetIndividual(combined);
            robotInstances.Add(rc);
        }
    }

    void FixedUpdate()
    {
        if (isCoolDown) { if (++coolDownStep > maxCoolDownSteps) { isCoolDown = false; coolDownStep = 0; } return; }
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
                }
                else if (activeIndividuals[i] && dynamicGeneLength)
                {
                    (float, string) ors = robotInstances[i].CheckOrientationSensor();
                    ExtendIndividual(i, ors.Item1, robotInstances[i].GetRoad());
                    float torque = torquePopulation[i][currentStep] * 500f;
                    float steer = steeringPopulation[i][currentStep] * 40f;
                    robotInstances[i].ManualApplyControl(torque, steer);
                }
            }
            currentStep++;

            for (int i = 0; i < populationSize; i++)
                if (activeIndividuals[i]) robotInstances[i].UpdateFitness(currentStep > 1000);
        }
        else if (!activeIndividuals.Contains(true))
        {
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
        int end = (int)( possibleValues.Count * 0.60f);
        int start = (int)(possibleValues.Count / 2);
        if(ors  <= -2f){
            end = (int) possibleValues.Count;
            start = (int)(possibleValues.Count * 0.9f);
        }
        float t = possibleValues[Random.Range(start, end)];

        // Debug.Log($"start: {start}, end: {end}, ors: {ors}, torque: {t}");

        int val1 = turn > 0 ? (int)(possibleValues.Count * 0.25f) : 0;
        int val2 = turn < 0 ? possibleValues.Count / 3 : possibleValues.Count;

        if(turn == 0 && ors <= -2f){
            val1 = (int)(possibleValues.Count * 0.25f);
            val2 = (int)(possibleValues.Count * 0.75f);
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
        }
    }

    void ResetGeneration()
    {
        activeIndividuals = new(new bool[populationSize]);
        for (int i = 0; i < populationSize; i++) activeIndividuals[i] = true;
        InitializeRobots();
    }

    void EvolveBothPopulations()
    {
        List<int> sorted = GetSortedIndices(steeringFitnessScores);
        Debug.Log($"Torque best: {Max(torqueFitnessScores)}, avg: {Average(torqueFitnessScores)}, generation: {currentGeneration}, geneLength: {currentGeneLength}");
        Debug.Log($"Steering best: {Max(steeringFitnessScores)}, avg: {Average(steeringFitnessScores)}, generation: {currentGeneration}, geneLength: {currentGeneLength}");

        CreateNewPopulationPair(ref torquePopulation, torqueFitnessScores, freezeIndexTorque,
                                ref steeringPopulation, steeringFitnessScores, freezeIndexSteering, sorted);
    }

    void CreateNewPopulationPair(ref List<List<float>> torquePop, List<float> torqueScores, int freezeTorque,
                                 ref List<List<float>> steerPop, List<float> steerScores, int freezeSteer, List<int> sorted)
    {
        int eliteCount = Mathf.Max(1, populationSize / 10);
        int poolSize = populationSize / 2;
        List<List<float>> newTorque = new();
        List<List<float>> newSteer = new();

        if (dynamicGeneLength)
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

        while (newTorque.Count < populationSize)
        {
            int idx1 = sorted[Random.Range(0, poolSize)];
            int idx2 = sorted[Random.Range(0, poolSize)];

            int? pt1 = null, pt2 = null;
            if (useSegmentCrossover && torquePop[idx1].Count > freezeTorque + 2)
            {
                pt1 = Random.Range(freezeTorque + 1, torquePop[idx1].Count - 2);
                pt2 = Random.Range(pt1.Value, torquePop[idx1].Count - 1);
            }

            SharedCrossover(torquePop[idx1], torquePop[idx2], out var c1T, out var c2T, freezeTorque, pt1, pt2);
            SharedCrossover(steerPop[idx1], steerPop[idx2], out var c1S, out var c2S, freezeSteer, pt1, pt2);

            Mutate(c1T, freezeTorque);
            Mutate(c2T, freezeTorque);
            Mutate(c1S, freezeSteer);
            Mutate(c2S, freezeSteer);

            ExtendToLength(c1T, currentGeneLength, torquePop[sorted[0]]);
            ExtendToLength(c2T, currentGeneLength, torquePop[sorted[0]]);
            ExtendToLength(c1S, currentGeneLength, steerPop[sorted[0]]);
            ExtendToLength(c2S, currentGeneLength, steerPop[sorted[0]]);

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

        if (useSegmentCrossover && p1.Count > freezeIdx + 2)
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
        else
        {
            for (int i = freezeIdx; i < p1.Count; i++)
            {
                if (Random.value < crossoverRate)
                {
                    float tmp = c1[i];
                    c1[i] = c2[i];
                    c2[i] = tmp;
                }
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
