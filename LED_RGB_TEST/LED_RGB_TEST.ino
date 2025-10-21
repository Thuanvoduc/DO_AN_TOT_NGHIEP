#define R1 14
#define G1 13

#define R2 12
#define G2 27

#define R3 26
#define G3 32

void setup() {
  // Cấu hình tất cả chân là OUTPUT
  pinMode(R1, OUTPUT);
  pinMode(G1, OUTPUT);
  pinMode(R2, OUTPUT);
  pinMode(G2, OUTPUT);
  pinMode(R3, OUTPUT);
  pinMode(G3, OUTPUT);
}

void loop() {
  // LED 1 - màu ngẫu nhiên
  analogWrite(R1, random(0,1023));
  analogWrite(G1, random(0,1023));

  // LED 2 - màu ngẫu nhiên
  analogWrite(R2, random(0,1023));
  analogWrite(G2, random(0,1023));

  // LED 3 - màu ngẫu nhiên
  analogWrite(R3, random(0,1023));
  analogWrite(G3, random(0,1023));

  delay(1000); // đổi màu sau mỗi giây
}
