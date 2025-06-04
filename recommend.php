<?php
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $target_calories = floatval($_POST['target_calories']);
    $meal_count = intval($_POST['meal_count']);

    $data = [
        'target_calories' => $target_calories,
        'meal_count' => $meal_count
    ];

    $options = [
        'http' => [
            'header'  => "Content-type: application/json",
            'method'  => 'POST',
            'content' => json_encode($data),
        ],
    ];
    $context  = stream_context_create($options);

    $result = file_get_contents('http://127.0.0.1:5000/recommend', false, $context);
    $response = json_decode($result, true);
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>Meal Plan Recommendation</title>
</head>
<body>
    <h1>Meal Plan Recommendation</h1>
    <form method="post">
        <label>Target Calories (kcal):</label><br>
        <input type="number" name="target_calories" required><br><br>

        <label>Number of Meals per Day:</label><br>
        <input type="number" name="meal_count" required min="1"><br><br>

        <input type="submit" value="Get Meal Plan">
    </form>

    <?php if (!empty($response) && !isset($response['error'])): ?>
        <h2>Recommended Meal Plan:</h2>
        <ul>
            <?php foreach ($response as $meal): ?>
                <li>
                    <?php echo htmlspecialchars($meal['meal_name']); ?> - 
                    Calories: <?php echo htmlspecialchars($meal['calories']); ?>
                   
                </li>
            <?php endforeach; ?>
        </ul>
    <?php elseif (!empty($response) && isset($response['error'])): ?>
        <p style="color:red;"><?php echo htmlspecialchars($response['error']); ?></p>
    <?php endif; ?>
</body>
</html>
