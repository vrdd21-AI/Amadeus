package com.example.amadeus;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class ChatRoomActivity extends AppCompatActivity {
    private TextView textView1;
    private TextView textView2;
    private TextView textView3;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_multiroom);

        textView1 = findViewById(R.id.lasttext1);
        textView2 = findViewById(R.id.lasttext2);
        textView3 = findViewById(R.id.lasttext3);
    }

    public void roomClicked(View view){

        Intent intent = new Intent(this, ChatActivity.class);
        if(view.getId() == R.id.room_1){
            intent.putExtra("name", "Kurisu");
            startActivityForResult(intent, 1);
        }
        else if(view.getId() == R.id.room_2){
            intent.putExtra("name", "Daru");
            startActivityForResult(intent, 2);
        }
        else if(view.getId() == R.id.room_3){
            intent.putExtra("name", "Mayushi");
            startActivityForResult(intent, 3);
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode == 10){
            Log.d("Test", "result code 10");
            setResult(10);
            finish();
        }

        try {
            if (requestCode == 1) {
                String last_text = data.getStringExtra("lasttext");
                textView1.setText(last_text);
            } else if (requestCode == 2) {
                String last_text = data.getStringExtra("lasttext");
                textView2.setText(last_text);
            } else if (requestCode == 3) {
                String last_text = data.getStringExtra("lasttext");
                textView3.setText(last_text);
            }
        } catch (Exception e){
            e.printStackTrace();
        }


    }
}
