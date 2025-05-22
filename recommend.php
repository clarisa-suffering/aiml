<?php

$data = [
    'User_ID' => $_POST['user_id'],
    'Calories' => floatval($_POST['Calories']),
    'Protein (g)' => floatval($_POST['Protein (g)']),
    'Carbohydrates (g)' => floatval($_POST['Carbohydrates (g)']),
    'Fat (g)' => floatval($_POST['Fat (g)']),
    'Fiber (g)' => floatval($_POST['Fiber (g)']),
    'Sugars (g)' => floatval($_POST['Sugars (g)']),
    'Sodium (mg)' => floatval($_POST['Sodium (mg)']),
    'Cholesterol (mg)' => floatval($_POST['Cholesterol (mg)']),
    'Water_Intake (ml)' => floatval($_POST['Water_Intake (ml)']),
    'Meal_Type' => $_POST['Meal_Type']  // Optional if you want to filter by meal type
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

echo "<h2>Top 3 Meal Recommendations:</h2><ul>";
foreach ($response as $meal) {
    echo "<li>{$meal['Food_Item']} ({$meal['Category']}, {$meal['Meal_Type']})</li>";
}
echo "</ul>";
?>
