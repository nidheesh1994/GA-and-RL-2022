using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading.Tasks;
using Unity.Barracuda;
using System.IO;

public class RobotAgent : Agent
{

    [SerializeField] private WheelCollider FLC;
    [SerializeField] private WheelCollider FRC;
    [SerializeField] private WheelCollider RLC;
    [SerializeField] private WheelCollider RRC;

    [SerializeField] private Transform FLT;
    [SerializeField] private Transform FRT;
    [SerializeField] private Transform RLT;
    [SerializeField] private Transform RRT;

    [SerializeField] private Transform FRS;
    [SerializeField] private Transform L1S;
    [SerializeField] private Transform L2S;
    [SerializeField] private Transform L3S;
    [SerializeField] private Transform R1S;
    [SerializeField] private Transform R2S;
    [SerializeField] private Transform R3S;
    [SerializeField] private Transform ORS;
    [SerializeField] private Transform Down;

    [Header("Movement Parameters")]
    public float motorTorque = 20f;

    public float brakeTorque = 20f;
    public float turnSpeed = 30f;

    [Header("Sensor Parameters")]
    public float sensorRange = 10f;
    public float obstacleDetectionDistance = 7f;
    public string roadMaterial = "MT_Road_01";

    [Header("Performance Tuning")]
    public float finalBrakeForce = -140f;
    public float steeringSmoothing = 5f;
    public float accelerationSmoothing = 2f;

    public float decelerationSmoothing = 10f;

    private float currentSteerAngle = 0f;
    private float currentMotorTorque = 0f;

    private float currentBrakeTorque = 0f;

    private GeneticAlgorithm ga;
    private int individualIndex;
    private float totalReward;
    private List<Vector2> currentIndividual;
    private bool isActive;
    public bool shouldRender = true; // Controls visual updates

    private float totalTorqueReward = 0f;
    private float totalSteeringReward = 0f;

    private bool finishingPointDetected = false;
    private Vector3 lastPosition;
    private float episodeTime = 0f;
    public NNModel onnxModelAsset;
    private Model runtimeModel;
    private IWorker worker;
    private StreamWriter csvWriter;
    private int currentStep = 0;
    private float reward = 0f;

    // private float torqueMean = 96.5890121459961f; first position
    // private float torqueStd = 130.80125427246094f;
    // private float steerMean = -0.21530772745609283f;
    // private float steerStd = 7.084796905517578f;

    // private float torqueMean = 90.01801300048828f; // Cobined postion and best 3
    // private float torqueStd = 124.83465576171875f;
    // private float steerMean = 0.861481249332428f;
    // private float steerStd = 6.110722064971924f;

    // private float torqueMean = 89.37027740478516f; // Cobined postion and best 5
    // private float torqueStd = 124.12162017822266f;
    // private float steerMean = 0.862391471862793f;
    // private float steerStd = 6.122593879699707f;

    // private float torqueMean = 92.2625961303711f; // All Cobined postion and best 3
    // private float torqueStd = 126.49352264404297f;
    // private float steerMean = 0.5075882077217102f;
    // private float steerStd = 6.6546711921691895f;

    // private float torqueMean = 67.46678161621094f; // Second only and best 3
    // private float torqueStd = 90.83110809326172f;
    // private float steerMean = 0.11064644157886505f;
    // private float steerStd = 6.895825386047363f;

    // private float torqueMean = 119.72089385986328f; // New map p1 and best 3
    // private float torqueStd = 151.7726593017578f;
    // private float steerMean = 1.3716881275177002f;
    // private float steerStd = 5.4518232345581055f;

    // private float torqueMean = 75.60594177246094f; // New map p2 and best 3 working on p2
    // private float torqueStd = 105.34881591796875f;
    // private float steerMean = -0.15299881994724274f;
    // private float steerStd = 6.910661220550537f;

    // private float torqueMean = 93.51426696777344f; // New map c12 and best 3  - 1200 epochs working from 2 points
    // private float torqueStd = 128.38467407226562f;
    // private float steerMean = 0.5537295937538147f;
    // private float steerStd = 6.386470794677734f;

    private float torqueMean = 99.20841979980469f; // New map c123 and best 3 - 1200 epochs
    private float torqueStd = 132.43568420410156f;
    private float steerMean = 0.9129691123962402f;
    private float steerStd = 6.612026214599609f;

    // private float torqueMean = 98.69376373291016f; // New map c1235 and best 3 - 1200 epochs
    // private float torqueStd = 132.77134704589844f;
    // private float steerMean = 0.7581171989440918f;
    // private float steerStd = 6.3204569816589355f;

    // private float torqueMean = 87.03598022460938f; // New map c1236 and best 3 - 1200 epochs
    // private float torqueStd = 120.14628601074219f;
    // private float steerMean = 0.4904853403568268f;
    // private float steerStd = 6.690590858459473f;

    public void Start()
    {
        // Reset the episode timer at the beginning of each episode
        episodeTime = 0f;

        runtimeModel = ModelLoader.Load(onnxModelAsset);
        // Create a worker to run the model
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, runtimeModel);

        // Stop the robot's velocity and angular velocity to prevent movement at the start
        Rigidbody rb = GetComponent<Rigidbody>();
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // Stop all wheels from rotating by setting motorTorque to zero
        FLC.motorTorque = 0f;
        FRC.motorTorque = 0f;
        RLC.motorTorque = 0f;
        RRC.motorTorque = 0f;

        // Reset wheel steer angles to prevent unintended turning
        FLC.steerAngle = 0f;
        FRC.steerAngle = 0f;

        string filePath = Path.Combine(Application.persistentDataPath, $"NN_Ooutput_{onnxModelAsset}.csv");
        csvWriter = new StreamWriter(filePath);
        csvWriter.WriteLine("Step,AppliedTorque,AppliedSteering,Fitness,Front,Left1,Left2,Left3,Right1,Right2,Right3,ORS,ORSZ,Speed");


        // transform.localPosition = new Vector3(34.56854f, 23.92629f, -243.2978f); // first position for GA working
        // transform.rotation = Quaternion.Euler(0f, 177.441f, -0.001f);


        transform.localPosition = new Vector3(-94.5086f, 39.55402f, -303.3212f); // sceond position for GA
        transform.rotation = Quaternion.Euler(-0.31f, 360.243f, 3.421f);

        // transform.localPosition = new Vector3(-93.75f, 34.68f, -242.83f); // third position for GA
        // transform.rotation = Quaternion.Euler(0.014f, 359.737f, 4.498f);

        // transform.localPosition = new Vector3(-94.4f, 39.8f, -184.5f); // fourth position for GA
        // transform.rotation = Quaternion.Euler(0.014f, 359.737f, 4.498f);

        // transform.localPosition = new Vector3(-94.4f, 39.8f, -184.5f); // fourth position reversed for GA
        // transform.rotation = Quaternion.Euler(0.014f, 180f, 4.498f);

        // transform.localPosition = new Vector3(-175.0308f, 35.79416f, -140.6013f); // fifth position for GA
        // transform.rotation = Quaternion.Euler(0.009f, 359.743f, 14.655f);

        // transform.localPosition = new Vector3(-175.0308f, 35.79416f, -140.6013f); // fifth position reversed for GA
        // transform.rotation = Quaternion.Euler(0.009f, 180f, 14.655f);

        // transform.localPosition = new Vector3(37.49f, 24.06f, -115.9f); // initial position for GA
        // transform.rotation = Quaternion.Euler(0.088f, 181.029f, -4.497f);


        lastPosition = transform.localPosition;

        // Set sensor orientations as defined
        SetSensorOrientations();
    }

    void FixedUpdate()
    {
        var sensorReadings = GetSensorData();
        float speed = Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);

        float[] inputData = new float[10];
        inputData[0] = sensorReadings["Front"].Item1;
        inputData[1] = sensorReadings["Left1"].Item1;
        inputData[2] = sensorReadings["Left2"].Item1;
        inputData[3] = sensorReadings["Left3"].Item1;
        inputData[4] = sensorReadings["Right1"].Item1;
        inputData[5] = sensorReadings["Right2"].Item1;
        inputData[6] = sensorReadings["Right3"].Item1;
        inputData[7] = sensorReadings["ORS"].Item1;
        inputData[8] = sensorReadings["ORSZ"].Item1;
        inputData[9] = speed;

        // Create the tensor with shape [batch, height, width, channels] = [1, 1, 1, 10]
        Tensor inputTensor = new Tensor(1, 1, 1, 10);

        // Populate the tensor with your input data
        for (int i = 0; i < 10; i++)
        {
            inputTensor[0, 0, 0, i] = inputData[i];
        }
        // Execute the model with the input tensor
        worker.Execute(inputTensor);

        // Retrieve the output tensor
        Tensor outputTensor = worker.PeekOutput();

        // Process the output as needed
        // For example, if the output is a single value:
        float normalizedTorque = outputTensor[0];
        float normalizedSteering = outputTensor[1];

        float actualTorque = normalizedTorque * torqueStd + torqueMean;
        float actualSteering = normalizedSteering * steerStd + steerMean;
        // float torque = outputTensor[0];
        // float steering = outputTensor[1];

        ManualApplyControl(actualTorque, actualSteering);

        float currentReward = HandleSteeringRewards(actualTorque, actualSteering);
        Debug.Log($"Current step: {currentStep}, current reward: {currentReward}");

        csvWriter.WriteLine($"{currentStep},{actualTorque},{actualSteering},{currentReward}, {sensorReadings["Front"].Item1}, {sensorReadings["Left1"].Item1}, {sensorReadings["Left2"].Item1}, {sensorReadings["Left3"].Item1}, {sensorReadings["Right1"].Item1}, {sensorReadings["Right2"].Item1}, {sensorReadings["Right3"].Item1}, {sensorReadings["ORS"].Item1}, {sensorReadings["ORSZ"].Item1}, {speed}");

        Debug.Log($"torque: {actualTorque}, steering: {actualSteering}");

        // Dispose of tensors to free resources
        inputTensor.Dispose();
        outputTensor.Dispose();
        currentStep++;
    }

    void OnApplicationQuit()
    {
        Debug.Log("Application ending after " + currentStep + " step");
        csvWriter.Close();
        Debug.Log($"NN data logged data logged to {Path.Combine(Application.persistentDataPath, $"NN_Ooutput_{onnxModelAsset}.csv")}");
    }

    public float HandleSteeringRewards(float steeringAngle, float torque)
    {
        float speed = Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);
        int road = GetRoad();

        if (speed > 0f)
        {
            reward += speed > 2f ? (speed < 6f ? 1f : -0.3f) : -1f;
        }
        else
        {
            reward -= 1f;
        }

        if (road == 1)
        {
            if (steeringAngle > 5f && steeringAngle < 30f)
            {
                // Debug.Log("Turning rewards adding");
                reward += steeringAngle > 5f ? 1f : 0f;
                reward += steeringAngle > 10f ? 2f : 0f;
                reward += steeringAngle > 15f ? 3f : 0f;
            }
            else
                reward += -5f;
        }
        else if (road == 2)
        {
            if (steeringAngle < -5f && steeringAngle > -15f)
            {
                // Debug.Log("Turning rewards adding");
                reward += steeringAngle < -5f ? 1f : 0f;
                reward += steeringAngle < -10f ? 2f : 0f;
                reward += steeringAngle < -15f ? 3f : 0f;
            }
            else
                reward += -5f;
        }

        if (road == 0 && steeringAngle > -10f && steeringAngle < 10f)
            reward += 1f;
        else
            reward += -1f;

        (float, string) ors = CheckOrientationSensor();
        if (ors.Item1 <= -2f)
        {
            if (torque >= 250f)
                reward += 3f;
            else if (torque <= 150f)
                reward += -1f;
        }
        reward += HandleEdgeDetection();

        // ✅ ➕ Add reward for distance covered since last frame
        float deltaDistance = Vector3.Distance(transform.localPosition, lastPosition);
        reward += deltaDistance * 100f; // 🔁 2f is the weight — adjust as needed
        // Debug.Log($"Last position: {lastPosition}, current position: {transform.position}, deltaDistance: {deltaDistance}");

        // Update last position
        lastPosition = transform.localPosition;

        return reward;
    }

    private float HandleEdgeDetection()
    {
        var sensorReadings = GetSensorData();

        if (sensorReadings["Left1"].Item2.StartsWith("ED") || sensorReadings["Right1"].Item2.StartsWith("ED") || sensorReadings["Front"].Item2.StartsWith("ED"))
        {
            // Debug.Log($"left1 : {sensorReadings["Left1"].Item1}, right1: {sensorReadings["Right1"].Item1}, Front: {sensorReadings["Front"].Item1}");
            return -3f;

        }

        return 5f;
    }

    public int GetRoad()
    {
        var sensorReadings = GetSensorData();
        string[] keysToCheck = { "Down", };

        foreach (var key in keysToCheck)
        {
            string hitObject = sensorReadings[key].Item2;
            // Debug.Log($"Hitobject: {hitObject}");

            if (hitObject.Contains("MT_Turn (1)") || hitObject.Contains("MT_Turn (2)") || hitObject.Contains("MT_Turn (8)"))
            {
                return 1;
            }
            else if (hitObject.Contains("MT_Turn (13)") || hitObject.Contains("MT_Turn (14)") || hitObject.Contains("MT_Turn (7)") || hitObject.Contains("MT_Turn (6)") || hitObject.Contains("MT_Turn (5)") || hitObject.Contains("MT_Turn (11)"))
            {
                return 2;
            }
            else if (hitObject.Contains("MT_Turn (15)") || hitObject.Contains("MT_Turn (9)") || hitObject.Contains("MT_Turn (12)") || hitObject.Contains("MT_Turn (10)"))
            {
                return 3;
            }
        }

        return 0;
    }

    private void SetSensorOrientations()
    {
        FRS.localRotation = Quaternion.Euler(8, 0, 0);
        L1S.localRotation = Quaternion.Euler(0, -15, 0);
        L2S.localRotation = Quaternion.Euler(0, -35, 0);
        L3S.localRotation = Quaternion.Euler(0, -90, 0);
        R1S.localRotation = Quaternion.Euler(0, 15, 0);
        R2S.localRotation = Quaternion.Euler(0, 35, 0);
        R3S.localRotation = Quaternion.Euler(0, 90, 0);
        Down.localRotation = Quaternion.Euler(90, 0, 0);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Ensure you're passing 10 floats here in the same order used in training
        // Example: 7 raycast distances + ORS + ORSZ + speed
        var sensorReadings = GetSensorData();
        float speed = Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);


        sensor.AddObservation(sensorReadings["Front"].Item1);
        sensor.AddObservation(sensorReadings["Left1"].Item1);
        sensor.AddObservation(sensorReadings["Left2"].Item1);
        sensor.AddObservation(sensorReadings["Left3"].Item1);
        sensor.AddObservation(sensorReadings["Right1"].Item1);
        sensor.AddObservation(sensorReadings["Right2"].Item1);
        sensor.AddObservation(sensorReadings["Right3"].Item1);
        sensor.AddObservation(sensorReadings["ORS"].Item1);
        sensor.AddObservation(sensorReadings["ORSZ"].Item1);
        sensor.AddObservation(speed);
    }

    public Dictionary<string, (float, string)> GetSensorData()
    {
        return new Dictionary<string, (float, string)>
        {
            { "Front", CheckSensor(FRS, true) },
            { "Left1", CheckSensor(L1S, true) },
            { "Left2", CheckSensor(L2S, true) },
            { "Left3", CheckSensor(L3S, true) },
            { "Right1", CheckSensor(R1S, true) },
            { "Right2", CheckSensor(R2S, true) },
            { "Right3", CheckSensor(R3S, true) },
            { "Down", CheckSensor(Down) },
            { "ORS", CheckOrientationSensor() },
            { "ORSZ", CheckOrientationSensorZ() }
        };
    }

    public (float, string) CheckOrientationSensor()
    {
        float xaw = transform.eulerAngles.x;
        float normalizedPitch = (xaw > 180) ? xaw - 360 : xaw;
        return (normalizedPitch, "OrientationX");
    }

    private (float, string) CheckOrientationSensorZ()
    {
        float zaw = transform.eulerAngles.z;
        float normalizedPitch = (zaw > 180) ? zaw - 360 : zaw;
        return (normalizedPitch, "OrientationZ");
    }

    private (float, string) CheckSensor(Transform sensor, bool draw = false)
    {
        RaycastHit hit;

        // Exclude 'Robot' layer
        int layerMask = ~LayerMask.GetMask("Robot");

        // ✅ Only draw if this is the Down sensor
        if (draw)
        {
            Debug.DrawRay(sensor.position, sensor.forward * sensorRange, Color.yellow);
        }

        if (Physics.Raycast(sensor.position, sensor.forward, out hit, sensorRange, layerMask))
        {
            Debug.DrawRay(sensor.position, sensor.forward * hit.distance, Color.red);
            return (hit.distance, hit.collider.gameObject.name);
        }
        return (sensorRange, "None");
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float motorInput = actions.ContinuousActions[0];
        float steerInput = actions.ContinuousActions[1];

        float motorTorque = motorInput * 500f;  // same as training
        float steerAngle = steerInput * 40f;

        // Apply to your robot
        ManualApplyControl(motorTorque, steerAngle);
    }

    public void ManualApplyControl(float torque, float steering)
    {
        ApplySteering(steering);
        ApplyMotorTorque(torque);
        // Debug.Log($"Torque: {torque}, steering: {currentSteerAngle}, car: {individualIndex}");
        UpdateWheelTransforms();
    }

    private void ApplySteering(float targetAngle)
    {
        // currentSteerAngle = Mathf.Lerp(currentSteerAngle, targetAngle, Time.deltaTime * steeringSmoothing);
        // Debug.Log($"currentSteerAngle: {currentSteerAngle}");
        currentSteerAngle = Mathf.Min(40f, targetAngle); ;
        FLC.steerAngle = currentSteerAngle;
        FRC.steerAngle = currentSteerAngle;
    }

    private void ApplyMotorTorque(float targetTorque)
    {
        // currentMotorTorque = Mathf.Lerp(currentMotorTorque, targetTorque, Time.deltaTime * accelerationSmoothing);
        currentMotorTorque = Mathf.Min(280f, targetTorque);
        if (finishingPointDetected && targetTorque == 0f)
            currentMotorTorque = 0;
        // Debug.Log($"currentMotorTorque: {currentMotorTorque}");
        FLC.motorTorque = currentMotorTorque;
        FRC.motorTorque = currentMotorTorque;
        RLC.motorTorque = currentMotorTorque;
        RRC.motorTorque = currentMotorTorque;
    }

    private void UpdateWheelTransforms()
    {
        UpdateWheelTransform(FLC, FLT);
        UpdateWheelTransform(FRC, FRT);
        UpdateWheelTransform(RLC, RLT);
        UpdateWheelTransform(RRC, RRT);
    }

    private void UpdateWheelTransform(WheelCollider collider, Transform wheelTransform)
    {
        Vector3 position;
        Quaternion rotation;
        collider.GetWorldPose(out position, out rotation);
        wheelTransform.position = position;
        wheelTransform.rotation = rotation;
    }
}
